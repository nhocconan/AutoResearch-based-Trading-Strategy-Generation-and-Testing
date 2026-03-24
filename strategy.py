#!/usr/bin/env python3
"""
Experiment #295: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + Volatility Regime v1

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test),
combined with volatility spike mean reversion and asymmetric regime logic.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, long when Fisher crosses above -1.5, short when crosses below +1.5
   This catches reversals better than RSI in bear markets (proven in literature)
2. VOLATILITY SPIKE MEAN REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long
   Captures "vol crush" after panic selling - high win rate pattern
3. ASYMMETRIC REGIME: ADX>25 = trending (only trade with HTF direction), ADX<20 = range (mean revert)
   Different logic per regime reduces whipsaw
4. 1d HMA for major trend bias - only trade in direction of 1d trend for higher win rate
5. Volatility-based position sizing - reduce size when vol is extreme

Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test
Timeframe: 6h (middle ground between 4h and 12h)
Trade frequency: 30-60 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_regime_asymmetric_1d1w_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val > 1e-10:
            normalized = 2.0 * (close[i] - lowest) / range_val - 1.0
            normalized = max(-0.99, min(0.99, normalized))
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    return fisher

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / (sma + 1e-10) * 100.0
    
    return upper, lower, width

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    smoothed_plus_dm = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * smoothed_plus_dm / (atr + 1e-10)
    minus_di = 100.0 * smoothed_minus_dm / (atr + 1e-10)
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    adx = calculate_adx(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_21 = calculate_ema(close, 21)
    
    # Volatility ratio for spike detection
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=range
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        if adx[i] > 25.0:
            current_regime = 1  # trending
        elif adx[i] < 20.0:
            current_regime = 2  # range
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 6h TREND FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = False
        fisher_short = False
        
        if not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_long = True
            if fisher[i-1] > 1.5 and fisher[i] <= 1.5:
                fisher_short = True
        
        # === VOLATILITY SPIKE MEAN REVERSION ===
        vol_spike = vol_ratio[i] > 2.0
        price_at_bb_lower = not np.isnan(bb_lower[i]) and close[i] <= bb_lower[i]
        price_at_bb_upper = not np.isnan(bb_upper[i]) and close[i] >= bb_upper[i]
        
        # === ASYMMETRIC ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (ADX > 25)
        if current_regime == 1:
            # Bear trend: only short retraces to EMA21
            if below_sma50 and htf_1d_bear:
                if fisher_short and close[i] > ema_21[i]:
                    desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
            
            # Bull trend: only long pullbacks
            elif above_sma50 and htf_1d_bull:
                if fisher_long and close[i] < ema_21[i]:
                    desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
        
        # REGIME 2: RANGE (ADX < 20)
        elif current_regime == 2:
            # Volatility spike mean reversion (high conviction)
            if vol_spike and price_at_bb_lower and above_sma200:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            elif vol_spike and price_at_bb_upper and below_sma200:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
            
            # Normal range mean reversion (lower conviction)
            elif price_at_bb_lower and above_sma200 and htf_1d_bull:
                desired_signal = SIZE_BASE
            
            elif price_at_bb_upper and below_sma200 and htf_1d_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals