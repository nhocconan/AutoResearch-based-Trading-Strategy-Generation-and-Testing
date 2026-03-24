#!/usr/bin/env python3
"""
Experiment #525: 15m Primary + 4h/1d HTF — Daily Pivot + HMA Trend + RSI Pullback

Hypothesis: 15m timeframe can work if we use HTF (4h/1d) for DIRECTION and 15m only for 
ENTRY TIMING. This gives HTF trade frequency with 15m execution precision. Key innovation:
Daily Pivot levels (CPR - Central Pivot Range) from 1d data provide natural support/resistance
where mean reversion occurs. Combined with 4h HMA trend bias and 15m RSI pullback entries,
this should capture trend continuations at value areas with minimal whipsaw.

Why 15m might work (learned from #517, #521 failures):
1. Use 4h HMA for trend DIRECTION (not 15m trend - too noisy)
2. Use 1d Pivot levels for ENTRY ZONES (natural S/R where price reacts)
3. Use 15m RSI(7) only for ENTRY TIMING (pullback within trend)
4. Session filter: 00-12 UTC (high volume, less fake breakouts)
5. Target 60-120 trades/year (strict confluence: 3+ signals must agree)
6. Size: 0.20 base (smaller than 4h strategies due to higher freq)

Strategy logic:
1. 1d Pivot: PP = (H+L+C)/3, BC = (H+L)/2, TC = PP + (PP-BC) [CPR range]
2. 4h HMA(21): trend bias (price > HMA = bull, price < HMA = bear)
3. 15m RSI(7): entry trigger (oversold <35 in uptrend, overbought >65 in downtrend)
4. 15m BB(20,2): position within band (lower band = long zone, upper = short zone)
5. 4h ADX(14): trend strength filter (ADX>20 = trend valid, skip if ADX<15)
6. Session: only entries 00-12 UTC (London/NY overlap for crypto)
7. Stoploss: 2.5x ATR(14) from entry, trail on profit

Regime-adaptive:
- TREND (4h ADX>25): Follow 4h HMA direction, enter on 15m RSI pullback
- RANGE (4h ADX<20): Mean revert at daily pivot levels (BC/TC)
- Size reduction in transition regimes

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pivot_hma_rsi_4h1d_session_v1"
timeframe = "15m"
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
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_daily_pivots(df_1d):
    """
    Calculate Daily Pivot levels from 1d data
    PP = (H + L + C) / 3
    BC = (H + L) / 2  (Bottom Central)
    TC = PP + (PP - BC)  (Top Central)
    CPR Range = BC to TC
    """
    n = len(df_1d)
    pp = np.zeros(n)
    bc = np.zeros(n)
    tc = np.zeros(n)
    
    for i in range(n):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        
        pp[i] = (h + l + c) / 3.0
        bc[i] = (h + l) / 2.0
        tc[i] = pp[i] + (pp[i] - bc[i])
    
    return pp, bc, tc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 4h ADX for trend strength
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
    # Calculate and align 1d Pivot levels
    pp_1d_raw, bc_1d_raw, tc_1d_raw = calculate_daily_pivots(df_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d_raw)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_lower[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_4h_aligned[i]) or np.isnan(pp_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # Parse hour from open_time (milliseconds since epoch)
        hour_utc = (open_time[i] // 3600000) % 24
        is_session = 0 <= hour_utc <= 12
        
        # === 4H TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # 4h ADX trend strength
        adx_trend = adx_4h_aligned[i] > 20.0  # Trend valid
        adx_strong = adx_4h_aligned[i] > 25.0  # Strong trend
        adx_weak = adx_4h_aligned[i] < 18.0   # Range/weak
        
        # === 1D PIVOT POSITION ===
        # Price relative to CPR (Central Pivot Range)
        in_cpr = bc_aligned[i] <= close[i] <= tc_aligned[i]
        above_cpr = close[i] > tc_aligned[i]
        below_cpr = close[i] < bc_aligned[i]
        
        # Near pivot levels (within 0.5% for mean reversion)
        pivot_range = max(tc_aligned[i] - bc_aligned[i], close[i] * 0.005)
        near_bc = abs(close[i] - bc_aligned[i]) < pivot_range * 0.5
        near_tc = abs(close[i] - tc_aligned[i]) < pivot_range * 0.5
        near_pp = abs(close[i] - pp_aligned[i]) < pivot_range * 0.3
        
        # === 15M RSI ENTRY TRIGGER ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_os = rsi[i] < 25.0
        rsi_extreme_ob = rsi[i] > 75.0
        rsi_recovering = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_declining = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === 15M BOLLINGER POSITION ===
        at_lower_bb = close[i] <= bb_lower[i] * 1.002
        at_upper_bb = close[i] >= bb_upper[i] * 0.998
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05  # Narrow bands
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG entries (need 4h bull + pullback + trigger)
        if htf_bull and is_session:
            confluence = 0
            
            # 4h trend confirmation
            if adx_trend:
                confluence += 1
            
            # Price at value area (near BC or lower BB)
            if near_bc or at_lower_bb:
                confluence += 1
            
            # RSI trigger (oversold or recovering)
            if rsi_oversold or (rsi_extreme_os and rsi_recovering):
                confluence += 1
            
            # Additional: above CPR breakout or near PP support
            if above_cpr or near_pp:
                confluence += 0.5
            
            if confluence >= 2.5:
                if adx_strong:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT entries (need 4h bear + rally + trigger)
        elif htf_bear and is_session:
            confluence = 0
            
            # 4h trend confirmation
            if adx_trend:
                confluence += 1
            
            # Price at resistance (near TC or upper BB)
            if near_tc or at_upper_bb:
                confluence += 1
            
            # RSI trigger (overbought or declining)
            if rsi_overbought or (rsi_extreme_ob and rsi_declining):
                confluence += 1
            
            # Additional: below CPR breakdown or near PP resistance
            if below_cpr or near_pp:
                confluence += 0.5
            
            if confluence >= 2.5:
                if adx_strong:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at CPR boundaries (ADX weak)
        if adx_weak and is_session:
            # Long at BC support with RSI oversold
            if near_bc and rsi_oversold and htf_bull:
                desired_signal = max(desired_signal, SIZE_BASE * 0.8)
            # Short at TC resistance with RSI overbought
            elif near_tc and rsi_overbought and htf_bear:
                desired_signal = min(desired_signal, -SIZE_BASE * 0.8)
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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