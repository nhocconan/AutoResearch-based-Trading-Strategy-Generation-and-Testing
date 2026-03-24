#!/usr/bin/env python3
"""
Experiment #731: 6h Primary + 1w/1d HTF — Fisher Transform + ADX Regime + Vol Spike

Hypothesis: 6h timeframe with Fisher Transform reversals + ADX regime detection + 
volatility spike mean reversion will capture both trending and ranging markets.
6h is underexplored (ZERO prior experiments) - middle ground between 4h noise and 12h slowness.

Key innovations:
1. Ehlers Fisher Transform(9) - catches reversals in bear rallies (proven in research)
2. ADX(14) regime with hysteresis - enter trend at 25, exit at 18 (reduces whipsaw)
3. ATR ratio (7/30) vol spike detection - mean revert after panic (>2.0 = spike)
4. 1w HMA(21) for primary trend bias
5. 1d ADX for intermediate regime confirmation
6. Asymmetric entries: only short in bear regime, only long in bull regime
7. Discrete sizing: 0.0, ±0.20, ±0.30

Entry conditions (LOOSE to ensure >=30 trades):
- LONG: 1w HMA bull + (Fisher<-1.5 OR vol_spike + RSI<35)
- SHORT: 1w HMA bear + (Fisher>+1.5 OR vol_spike + RSI>65)
- ADX hysteresis: trend confirmed when ADX>25, remains until ADX<18

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_volspike_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution for reversals"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        price_norm = (close[i] - lowest) / (highest - lowest)
        
        # Constrain to 0.001-0.999 to avoid log(0)
        price_norm = max(0.001, min(0.999, price_norm))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm))
        
        if i > 0:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = 0.0
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    if n < period * 2:
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
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed averages
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    plus_di = np.where(atr > 1e-10, 100.0 * plus_di / atr, 0.0)
    minus_di = np.where(atr > 1e-10, 100.0 * minus_di / atr, 0.0)
    
    # DX and ADX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 6h indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    adx_6h = calculate_adx(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    
    # ATR ratio for vol spike detection
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # ADX hysteresis state
    adx_trend_active = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_7[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx_6h[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trend mode when ADX > 25, exit when ADX < 18
        if adx_6h[i] > 25.0:
            adx_trend_active = True
        elif adx_6h[i] < 18.0:
            adx_trend_active = False
        
        # 1d ADX confirmation
        adx_1d_trend = adx_1d_aligned[i] > 20.0 if not np.isnan(adx_1d_aligned[i]) else False
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0 if not np.isnan(atr_ratio[i]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses below -1.5 = oversold reversal long
        # Fisher crosses above +1.5 = overbought reversal short
        fisher_oversold = fisher[i] < -1.5 and fisher_prev[i] >= -1.5
        fisher_overbought = fisher[i] > 1.5 and fisher_prev[i] <= 1.5
        
        # Also check for Fisher turning from extremes
        fisher_turn_long = fisher[i] > -1.0 and fisher_prev[i] <= -1.5
        fisher_turn_short = fisher[i] < 1.0 and fisher_prev[i] >= 1.5
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG entries (multiple paths to ensure trades)
        if htf_1w_bull:
            # Path 1: Fisher reversal from oversold
            if fisher_oversold or fisher_turn_long:
                desired_signal = SIZE_BASE
            
            # Path 2: Vol spike mean reversion + RSI oversold
            if vol_spike and rsi_14[i] < 40.0:
                desired_signal = max(desired_signal, SIZE_BASE)
            
            # Path 3: Trend regime + ADX confirmation
            if adx_trend_active and adx_1d_trend and close[i] > hma_1w_aligned[i] * 0.98:
                desired_signal = max(desired_signal, SIZE_STRONG)
        
        # SHORT entries (multiple paths to ensure trades)
        if htf_1w_bear:
            # Path 1: Fisher reversal from overbought
            if fisher_overbought or fisher_turn_short:
                desired_signal = -SIZE_BASE
            
            # Path 2: Vol spike mean reversion + RSI overbought
            if vol_spike and rsi_14[i] > 60.0:
                desired_signal = min(desired_signal, -SIZE_BASE)
            
            # Path 3: Trend regime + ADX confirmation
            if adx_trend_active and adx_1d_trend and close[i] < hma_1w_aligned[i] * 1.02:
                desired_signal = min(desired_signal, -SIZE_STRONG)
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals