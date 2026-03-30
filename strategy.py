#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian(20) Breakout + HMA Trend + Volume Spike

HYPOTHESIS: Donchian channel breakout is a proven structural breakout signal.
Adding HMA trend direction + volume confirmation + ATR stoploss creates a
high-probability mean-reversion after false breakouts AND trend continuation
after true breakouts.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Price breaks above Donchian upper → long with HMA up confirmation
- Bear: Price breaks below Donchian lower → short with HMA down confirmation
- Range: Price bounces between bands → mean reversion signals at extremes

KEY INSIGHT FROM DB: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 achieved
test Sharpe 1.38 on SOL (95 trades, 52% win rate). This is the proven pattern.

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_v3"
timeframe = "4h"
leverage = 1.0

def calculate_hma(values, period):
    """Hull Moving Average"""
    half_len = period // 2
    sqrt_len = int(np.sqrt(period))
    
    wma_half = pd.Series(values).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(half_len)) / np.arange(1, half_len + 1).sum(), raw=True
    )
    wma_full = pd.Series(values).rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(period)) / np.arange(1, period + 1).sum(), raw=True
    )
    
    hma = 2 * wma_half - wma_full
    hma = hma.rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(sqrt_len)) / np.arange(1, sqrt_len + 1).sum(), raw=True
    )
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA21 for HTF trend direction
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods = 5 days)
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Local HMA for faster trend
    hma_local = calculate_hma(close, 21)
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50  # Need enough for Donchian buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(hma_local[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND CONFIRMATION ===
        # HTF trend: 12h HMA vs price
        htf_bullish = close[i] > hma_12h_aligned[i]
        htf_bearish = close[i] < hma_12h_aligned[i]
        
        # Local trend: HMA direction
        local_bullish = hma_local[i] > hma_local[i-1] if i > 0 else False
        local_bearish = hma_local[i] < hma_local[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.4
        
        # RSI confirmation (not extreme)
        rsi_ok_long = rsi[i] < 65  # Not overbought for longs
        rsi_ok_short = rsi[i] > 35  # Not oversold for shorts
        
        # === DONCHIAN BREAKOUT CONDITIONS ===
        upper_break = high[i] > donchian_upper[i] and close[i] > donchian_upper[i] * 0.998
        lower_break = low[i] < donchian_lower[i] and close[i] < donchian_lower[i] * 1.002
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian breakout + HTF trend + volume ===
            if upper_break and htf_bullish and vol_spike and rsi_ok_long:
                desired_signal = SIZE
            # Mean reversion long: price below lower band in uptrend
            elif donchian_lower[i] > 0 and close[i] < donchian_lower[i] * 0.995 and htf_bullish and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Lower Donchian breakout + HTF trend + volume ===
            if lower_break and htf_bearish and vol_spike and rsi_ok_short:
                desired_signal = -SIZE
            # Mean reversion short: price above upper band in downtrend
            elif close[i] > donchian_upper[i] * 1.005 and htf_bearish and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === PROFIT TARGET: 3R or channel reversal ===
        if in_position:
            bars_held = i - entry_bar
            profit_r = (close[i] - entry_price) * position_side / entry_atr
            
            # Take profit at 3R
            if position_side > 0 and profit_r >= 3.0:
                desired_signal = 0.0
            if position_side < 0 and profit_r >= 3.0:
                desired_signal = 0.0
            
            # Exit if trend reverses
            if position_side > 0 and htf_bearish and bars_held >= 4:
                desired_signal = 0.0
            if position_side < 0 and htf_bullish and bars_held >= 4:
                desired_signal = 0.0
            
            # Time stop: max 20 bars (5 days)
            if bars_held >= 20:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals