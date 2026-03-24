#!/usr/bin/env python3
"""
Experiment #623: 6h Primary + 1d/1w HTF — Donchian Breakout + ADX Trend Strength + Volume Confirm

Hypothesis: 6h timeframe needs trend-following with strong filters to avoid choppy whipsaws.
Donchian(20) breakouts work well in trending markets (2021 bull, 2023 recovery) but fail
in chop. Adding ADX(14)>25 filter ensures we only trade when trend has momentum.
Volume confirmation (taker_buy_ratio > 0.55) filters false breakouts.

Key improvements over #620 (Fisher, Sharpe=-0.11):
1. Donchian breakout instead of Fisher reversals - better for trending periods
2. ADX(14) > 25 confirms trend strength before entry
3. Volume confirmation via taker_buy_volume ratio
4. 5-bar cooldown after exit prevents re-entry whipsaws
5. Wider stoploss (3*ATR) reduces premature exits
6. Conservative size (0.20-0.25) controls drawdown

Strategy logic:
1. 1w HMA(21) = macro trend bias (very slow, direction only)
2. 1d HMA(21) = medium trend bias (entry direction filter)
3. 6h Donchian(20) = breakout levels
4. 6h ADX(14) = trend strength filter (>25 = trade, <20 = wait)
5. 6h ATR(14) = volatility + stoploss (3*ATR)
6. Volume confirm: taker_buy_volume / volume > 0.55 for longs, <0.45 for shorts

Regime-adaptive:
- ADX>25 + HTF aligned = trend breakout (full size)
- ADX<20 = no trades (chop filter)
- ADX 20-25 = half size entries only

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_adx_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0.0)
        minus_dm[i] = max(low[i-1] - low[i], 0.0)
        if plus_dm[i] > minus_dm[i] and plus_dm[i] > 0:
            minus_dm[i] = 0.0
        elif minus_dm[i] > plus_dm[i] and minus_dm[i] > 0:
            plus_dm[i] = 0.0
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.full(n, np.nan)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is EMA of DX
    dx_series = pd.Series(dx)
    adx = dx_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout levels
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
    # Volume ratio (taker buy / total volume)
    taker_ratio = np.zeros(n)
    taker_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Cooldown counter (bars since last exit)
    cooldown = 0
    last_exit_side = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            cooldown = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(adx[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            cooldown = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            cooldown = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === 6H LOCAL TREND ===
        local_bull = close[i] > hma_6h[i]
        local_bear = close[i] < hma_6h[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 25.0   # Trending - full size entries
        adx_moderate = 20.0 < adx[i] <= 25.0  # Moderate - half size only
        adx_weak = adx[i] <= 20.0    # Choppy - no entries
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm_long = taker_ratio[i] > 0.55
        vol_confirm_short = taker_ratio[i] < 0.45
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade when ADX indicates trend (avoid chop)
        if adx_strong or adx_moderate:
            size_multiplier = 1.0 if adx_strong else 0.8
            
            # LONG: HTF bull + breakout + volume confirm
            if htf_bull and local_bull and breakout_long and vol_confirm_long:
                desired_signal = SIZE_STRONG * size_multiplier
            # SHORT: HTF bear + breakout + volume confirm
            elif htf_bear and local_bear and breakout_short and vol_confirm_short:
                desired_signal = -SIZE_STRONG * size_multiplier
            # Local trend breakout with HTF neutral (smaller size)
            elif not htf_bull and not htf_bear:
                if local_bull and breakout_long and vol_confirm_long:
                    desired_signal = SIZE_BASE * size_multiplier
                elif local_bear and breakout_short and vol_confirm_short:
                    desired_signal = -SIZE_BASE * size_multiplier
        
        # === COOLDOWN CHECK (prevent re-entry whipsaws) ===
        if cooldown < 5 and last_exit_side != 0:
            # Block re-entry in same direction for 5 bars after exit
            if np.sign(desired_signal) == last_exit_side:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (3x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                if in_position and np.sign(final_signal) != position_side:
                    # Closing position triggers cooldown
                    cooldown = 0
                    last_exit_side = position_side
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                # Exiting position
                cooldown = 0
                last_exit_side = position_side
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        # Increment cooldown if not in position
        if not in_position:
            cooldown += 1
        
        signals[i] = final_signal
    
    return signals