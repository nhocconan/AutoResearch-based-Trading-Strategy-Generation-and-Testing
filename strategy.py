#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + Weekly Trend + Volume Confirmation

HYPOTHESIS:
- Weekly (1w) EMA21 for trend direction (captures macro structure)
- Daily Donchian(20) breakout for entry (proven price channel from DB winners)
- Volume spike confirmation (2.0x) to filter false breakouts
- Choppiness filter (CHOP < 55) to avoid ranging markets

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: Weekly EMA up + daily breakout up + vol spike + CHOP < 55 → strong longs
- Bear: Weekly EMA down + daily breakout down + vol spike + CHOP < 55 → strong shorts
- Range: CHOP > 61.8 → skip entirely (avoids whipsaws)
- ATR stoploss scales with volatility (survives 2022 crash)

KEY DIFFERENCE FROM #006:
- #006: 4h trailing channel + CHOP < 45 = 145 trades (Sharpe 0.271)
- #010: 1d Donchian + 1w trend + CHOP < 55 = ~60-100 trades (more selective)

TARGET: 60-100 total trades over 4 years (15-25/year) on 1d timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_trend_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging → SKIP
    CHOP < 55 = trending → ENTER
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1w HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA21 for macro trend direction
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channel (20-day)
    channel_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    channel_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # 20 for channel + 20 for vol MA + 14 for CHOP
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(channel_up[i]) or np.isnan(channel_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 55  # trending but not too strict
        
        # === WEEKLY TREND DIRECTION ===
        weekly_trend_up = close[i] > weekly_ema_aligned[i]
        weekly_trend_down = close[i] < weekly_ema_aligned[i]
        
        # === VOLUME CONFIRMATION (2.0x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks ABOVE previous channel high
        # Short: price breaks BELOW previous channel low
        prev_channel_up = channel_up[i - 1]
        prev_channel_lo = channel_lo[i - 1]
        
        breakout_up = close[i] > prev_channel_up
        breakout_down = close[i] < prev_channel_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + breakout up + weekly trend up + volume spike ===
            if breakout_up and weekly_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Trending + breakout down + weekly trend down + volume spike ===
            if breakout_down and weekly_trend_down and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Long: stop if price falls 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips to down
                if weekly_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short: stop if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips to up
                if weekly_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars (1d timeframe, less fee impact) ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals