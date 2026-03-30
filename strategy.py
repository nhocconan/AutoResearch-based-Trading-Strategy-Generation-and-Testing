#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian + HMA(16) + RSI + Volume + 1d EMA Trend

HYPOTHESIS: Combine proven patterns from DB:
- Donchian(20) breakout for structure (confirmed winners on SOLUSDT)
- HMA(16) for trend direction (stronger signal than SMA)
- RSI(14) for momentum confirmation (used in 1.38 and 1.46 Sharpe winners)
- Volume spike confirmation (mandatory in all winners)
- ATR(14) trailing stop (proven risk management)
- 1d EMA(55) as HTF trend filter

WHY IT WORKS: Donchian breakout catches major moves when all filters align.
HTF trend keeps us on the right side. Volume confirms institutional interest.
RSI adds momentum confirmation without over-filtering.

TARGET: 100-200 total trades over 4 years (25-50/year).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(values, period):
    """Hull Moving Average"""
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return pd.Series(series).rolling(window=length, min_periods=length).apply(
            lambda x: np.sum(x * weights) / np.sum(weights), raw=True
        ).values
    
    hma = wma(values, sqrt_length) * 2 - wma(values, half_length)
    return hma

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA55 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Pre-compute all indicators before loop ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # HMA(16) for local trend
    hma_16 = calculate_hma(close, 16)
    
    # Donchian channels (20 periods)
    period_donchian = 20
    donchian_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    donchian_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals array
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    stop_price = 0.0
    
    warmup = 100  # Buffer for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_14[i]) or np.isnan(hma_16[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTER CONDITIONS ===
        # 1d EMA trend (HTF filter)
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Local HMA trend
        hma_trending_up = hma_16[i] > hma_16[i-1]
        hma_trending_down = hma_16[i] < hma_16[i-1]
        
        # Volume confirmation (1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # RSI momentum
        rsi_value = rsi_14[i]
        rsi_bullish = rsi_value > 50
        rsi_bearish = rsi_value < 50
        
        # Donchian values
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout + HTF trend + momentum ===
            long_conditions = (
                price_above_1d_ema and      # HTF uptrend
                vol_spike and                # Volume confirmation
                close[i] > d_high and       # Breakout above upper band
                rsi_bullish and              # RSI confirms
                hma_trending_up             # Local HMA agrees
            )
            
            if long_conditions:
                desired_signal = SIZE
            
            # === SHORT: Donchian breakdown + HTF trend + momentum ===
            short_conditions = (
                price_below_1d_ema and      # HTF downtrend
                vol_spike and                # Volume confirmation
                close[i] < d_low and        # Breakdown below lower band
                rsi_bearish and             # RSI confirms
                hma_trending_down           # Local HMA agrees
            )
            
            if short_conditions:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === TAKE PROFIT (2R or mid-band after min hold) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:  # Min hold = 3 bars (12h)
            if position_side > 0:
                profit_target = entry_price + 2.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = 0.0
                elif close[i] >= d_mid:  # Price reverted to mid-band
                    desired_signal = 0.0
            
            elif position_side < 0:
                profit_target = entry_price - 2.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = 0.0
                elif close[i] <= d_mid:  # Price reverted to mid-band
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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
        
        signals[i] = desired_signal
    
    return signals