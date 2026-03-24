#!/usr/bin/env python3
"""
Experiment #423: 6h Primary + 1d/1w HTF — Fisher Transform + ADX Regime v1

Hypothesis: Previous 6h strategies failed due to:
1. Weekly pivot levels (too rigid, failed in #411, #415, #420)
2. CRSI mean reversion (negative Sharpe across multiple attempts)
3. Overly complex entry conditions (0 trades on lower TF attempts)

NEW APPROACH:
- Fisher Transform for entry timing (proven reversal catcher in bear markets)
- Weekly HMA(21) for major trend bias (smoother than pivots)
- Daily HMA(21) for intermediate confirmation
- ADX regime detection (trending vs ranging)
- Simpler entries: max 3 confluence conditions

Why 6h is unique:
- Captures multi-day swings without 12h lag
- 30-60 trades/year target (sweet spot for fee vs opportunity)
- 1w HTF provides major bias, 1d provides timing confirmation

Entry Logic:
- Trending (ADX>25): Fisher cross + HMA alignment + 1w bias
- Ranging (ADX<20): Fisher extremes + 1d HMA mean reversion

Position sizing: 0.25 base, 0.30 with full HTF alignment
Stoploss: 2.5x ATR(14) from entry
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_regime_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching reversals than RSI in bear/range markets
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1 range
    fisher_input = np.zeros(n)
    fisher_input[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            fisher_input[i] = 0.66 * ((typical[i] - lowest) / price_range - 0.5) + 0.67 * fisher_input[i-1]
            # Clamp to -0.99 to +0.99
            fisher_input[i] = max(-0.99, min(0.99, fisher_input[i]))
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_trigger = np.zeros(n)
    fisher_trigger[:] = np.nan
    
    for i in range(period, n):
        if abs(fisher_input[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + fisher_input[i]) / (1.0 - fisher_input[i]))
            fisher_trigger[i] = 0.5 * np.log((1.0 + fisher_input[i-1]) / (1.0 - fisher_input[i-1])) if i > period else fisher[i]
    
    return fisher, fisher_trigger

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=ranging
    
    # Position tracking for stoploss
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
        
        if np.isnan(hma_6h[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with ADX ===
        # Trending: ADX > 25
        # Ranging: ADX < 20
        # Otherwise: use previous regime (hysteresis)
        
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        if is_trending:
            current_regime = 1
        elif is_ranging:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1w and 1d) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i]) and not np.isnan(hma_6h_fast[i-1]):
            if not np.isnan(hma_6h[i]) and not np.isnan(hma_6h[i-1]):
                if hma_6h_fast[i-1] <= hma_6h[i-1] and hma_6h_fast[i] > hma_6h[i]:
                    hma_cross_long = True
                if hma_6h_fast[i-1] >= hma_6h[i-1] and hma_6h_fast[i] < hma_6h[i]:
                    hma_cross_short = True
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = False
        fisher_short = False
        
        if not np.isnan(fisher[i]) and not np.isnan(fisher_trigger[i]):
            if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
                # Fisher cross above -1.5 from below
                if fisher_trigger[i-1] < -1.5 and fisher_trigger[i] >= -1.5:
                    fisher_long = True
                # Fisher cross below +1.5 from above
                if fisher_trigger[i-1] > 1.5 and fisher_trigger[i] <= 1.5:
                    fisher_short = True
        
        # === FISHER EXTREMES (for ranging regime) ===
        fisher_oversold = not np.isnan(fisher[i]) and fisher[i] < -2.0
        fisher_overbought = not np.isnan(fisher[i]) and fisher[i] > 2.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI CONFIRMATION (loose thresholds for more trades) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (ADX > 25)
        # Entry: Fisher reversal + HMA alignment + HTF bias
        if current_regime == 1:
            # Long: 1w bull + 1d bull + Fisher long cross
            if htf_1w_bull and htf_1d_bull and fisher_long:
                desired_signal = SIZE_STRONG
            
            # Alternative long: HMA cross + 1d bull + RSI confirm
            elif hma_cross_long and htf_1d_bull and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # Short: 1w bear + 1d bear + Fisher short cross
            elif htf_1w_bear and htf_1d_bear and fisher_short:
                desired_signal = -SIZE_STRONG
            
            # Alternative short: HMA cross + 1d bear + RSI confirm
            elif hma_cross_short and htf_1d_bear and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: RANGING (ADX < 20)
        # Entry: Fisher extremes + mean reversion toward 1d HMA
        elif current_regime == 2:
            # Long: Fisher oversold + above SMA200 + price below 1d HMA (pullback)
            if fisher_oversold and above_sma200 and close[i] < hma_1d_aligned[i]:
                desired_signal = SIZE_BASE
            
            # Short: Fisher overbought + below SMA200 + price above 1d HMA (rally)
            elif fisher_overbought and below_sma200 and close[i] > hma_1d_aligned[i]:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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