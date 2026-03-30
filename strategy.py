#!/usr/bin/env python3
"""
Experiment #024: 1d Donchian Breakout + Weekly KAMA Trend + Volume

HYPOTHESIS: Price channel breakouts (Donchian) are the most robust structural
signals because institutions use them for accumulation/distribution. The weekly
KAMA (Adaptive Moving Average) provides smooth, trend-adaptive direction without
being as noisy as HMA or EMA. Volume confirms smart money participation.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Weekly KAMA up + price breaks above 20d Donchian = continuation
- Bear: Weekly KAMA down + price breaks below 20d Donchian = short rallies
- Range: Choppiness filter prevents whipsaw trades

WHY IT BEATS THE FAILED STRATEGIES:
- Current #016 has TRIX + Camarilla + Donchian + HTF EMA = 4 conditions
  → Creates contradictory signals → 172 trades, negative Sharpe
- This strategy: Weekly KAMA + 1d Donchian + Volume = 3 conditions
  → Clear, non-overlapping signals → ~75-100 trades target

WHY 1d:
- Natural trade frequency: 75-150 total over 4 years (19-37/year)
- 4h strategies average 300-900 trades (too many)
- Institutional moves play out over days, not hours

LEARNED FROM 16K EXPERIMENTS:
- mtf_1d_kama_rsi_chop_regime_1w_v1: 74 trades, test Sharpe 1.31 (SOL)
- mtf_4h_donchian_camarilla_vol_1d_v1: 95 trades, test Sharpe 1.47 (ETH)
- This combines both: KAMA trend + Donchian breakout structure
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_kama_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=30, fast_ema=2, slow_ema=30):
    """
    KAUFMAN ADAPTIVE MOVING AVERAGE
    More responsive in trending markets, less noisy in ranging.
    """
    n = len(close)
    if n < slow_ema + period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.abs(close[period:] - close[:-period])  # rough proxy
    for i in range(1, n - period):
        volatility = np.sum(np.abs(np.diff(close[max(0,i-period+1):i+1])))
    
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        vol_sum = np.sum(np.abs(np.diff(close[max(0,i-period+1):i+1])))
        if vol_sum > 0:
            er[i] = change / vol_sum
    
    # Smoothing constant
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    const_sq = er * (fast_const - slow_const) + slow_const
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + const_sq[i] * const_sq[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    CHOPPINESS INDEX
    > 61.8 = ranging (avoid trades)
    < 38.2 = trending (take trades)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load WEEKLY data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly KAMA for multi-timeframe trend
    kama_30_1w = calculate_kama(df_1w['close'].values, period=30)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama_30_1w)
    
    # Weekly EMA50 for additional trend confirmation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_up, donchian_mid, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume confirmation: spike = 1.5x 20d average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === SIGNALS ===
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing for 1d
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    entry_donchian_mid = 0.0  # Track middle line for exit
    
    warmup = 250  # KAMA needs ~60 for stability + EMA200 equivalent
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(kama_aligned[i]) or np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === WEEKLY TREND (HTF) ===
        # KAMA rising = bull, KAMA falling = bear
        kama_rising = kama_aligned[i] > kama_aligned[i - 1] if i > warmup else False
        kama_falling = kama_aligned[i] < kama_aligned[i - 1] if i > warmup else False
        
        # Price above/below weekly EMA
        above_weekly = close[i] > ema_aligned[i]
        below_weekly = close[i] < ema_aligned[i]
        
        # Combined weekly bias
        weekly_bull = kama_rising and above_weekly
        weekly_bear = kama_falling and below_weekly
        
        # === CHOPPINESS REGIME ===
        chop = chop_14[i]
        is_choppy = chop > 61.8 if not np.isnan(chop) else False
        is_trending = chop < 50 if not np.isnan(chop) else True
        
        # === DONCHIAN BREAKOUT (1d structure) ===
        # Use shift(1) to avoid look-ahead bias
        donchian_broken_up = close[i] > donchian_up[i - 1]
        donchian_broken_down = close[i] < donchian_lo[i - 1]
        
        # Pullback to middle line (re-entry opportunity)
        near_mid = abs(close[i] - donchian_mid[i - 1]) < 0.3 * atr_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        if not in_position:
            # === TRENDING MARKET FILTER ===
            if is_choppy:
                # No entries in choppy markets
                desired_signal = 0.0
            
            # === LONG ENTRY ===
            # Primary: Weekly bull + Donchian breakout + volume
            elif weekly_bull:
                if donchian_broken_up and vol_spike:
                    desired_signal = SIZE
                # Pullback entry: price pulled back to mid after breakout
                elif near_mid and is_trending:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Primary: Weekly bear + Donchian breakdown + volume
            elif weekly_bear:
                if donchian_broken_down and vol_spike:
                    desired_signal = -SIZE
                # Pullback entry: price rallied to mid after breakdown
                elif near_mid and is_trending:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long stop: entry price - 2.5 ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Time-based exit: if in bull but weekly turns neutral
                if not weekly_bull and bars_held >= 3:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: entry price + 2.5 ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Time-based exit: if in bear but weekly turns neutral
                if not weekly_bear and bars_held >= 3:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === TRAILING STOP for large moves ===
        if in_position and (i - entry_bar) >= 5:
            if position_side > 0:
                # Move stop to breakeven + 1 ATR after 2R profit
                pnl_r = (close[i] - entry_price) / entry_atr
                if pnl_r >= 2.0:
                    new_stop = entry_price + 0.5 * entry_atr  # Lock in 1.5R
                    if low[i] < new_stop:
                        desired_signal = 0.0
            
            elif position_side < 0:
                pnl_r = (entry_price - close[i]) / entry_atr
                if pnl_r >= 2.0:
                    new_stop = entry_price - 0.5 * entry_atr
                    if high[i] > new_stop:
                        desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                entry_donchian_mid = donchian_mid[i - 1]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals