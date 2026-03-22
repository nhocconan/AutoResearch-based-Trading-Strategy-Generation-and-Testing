#!/usr/bin/env python3
"""
Experiment #471: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX + RSI

Hypothesis: After analyzing 470 failed experiments, clear pattern emerges:
1. KAMA (Kaufman Adaptive MA) outperforms HMA/EMA in crypto's variable volatility
2. ADX > 20 (not 25+) confirms trend without over-filtering (more trades)
3. Simple RSI(14) extremes work better than CRSI for 4h timeframe
4. 1w HTF provides major cycle bias without whipsaw
5. Fewer conflicting filters = more trades = better statistical significance

Why this might beat current best (Sharpe=0.435):
- KAMA adapts ER (Efficiency Ratio) to market noise automatically
- ADX threshold relaxed to 20 for more trade opportunities
- 1w major trend filter prevents counter-trend trades in strong moves
- Asymmetric sizing protects in bear markets (0.30 long, 0.25 short)
- ATR 2.5x trailing stop limits drawdown in 2022-style crashes

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_1d1w_trend_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    
    ER = |Net Change| / Sum of |Individual Changes| over period
    SC = (ER * (fast SC - slow SC) + slow SC)^2
    KAMA = prior KAMA + SC * (price - prior KAMA)
    
    Best for crypto's variable volatility regimes.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Net change over period
    net_change = np.abs(close_s.diff(period))
    
    # Sum of individual changes over period
    individual_changes = np.abs(close_s.diff())
    sum_changes = individual_changes.rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio (ER)
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]  # initialize
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging (we use 20 for more trades)
    """
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_smooth = plus_dm_s.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm_s.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-10))
    
    # DX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators (intermediate trend)
    kama_1d_20 = calculate_kama(df_1d['close'].values, period=20)
    
    # Calculate 1w HTF indicators (major cycle bias)
    kama_1w_10 = calculate_kama(df_1w['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    kama_1w_10_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_10)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_10 = calculate_kama(close, period=10)
    kama_4h_30 = calculate_kama(close, period=30)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1d_20_aligned[i]) or np.isnan(kama_1w_10_aligned[i]):
            continue
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_30[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 1W MAJOR CYCLE BIAS (primary direction filter) ===
        # Price above 1w KAMA = bull cycle (favor longs)
        # Price below 1w KAMA = bear cycle (favor shorts)
        bull_cycle = close[i] > kama_1w_10_aligned[i]
        bear_cycle = close[i] < kama_1w_10_aligned[i]
        
        # === 1D INTERMEDIATE TREND (confirmation) ===
        bull_1d = close[i] > kama_1d_20_aligned[i]
        bear_1d = close[i] < kama_1d_20_aligned[i]
        
        # === 4H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_4h_10[i] > kama_4h_30[i]
        kama_bearish = kama_4h_10[i] < kama_4h_30[i]
        
        # === ADX TREND STRENGTH (relaxed threshold for more trades) ===
        adx_trending = adx_14[i] > 20.0  # relaxed from 25
        adx_strong = adx_14[i] > 30.0
        plus_di_above = plus_di[i] > minus_di[i]
        minus_di_above = minus_di[i] > plus_di[i]
        
        # === RSI ENTRY TIMING (simple extremes) ===
        rsi_oversold = rsi_14[i] < 35.0  # relaxed from 30 for more trades
        rsi_overbought = rsi_14[i] > 65.0  # relaxed from 70 for more trades
        rsi_neutral = 40.0 < rsi_14[i] < 60.0
        
        # === SMA200 FILTER (long-term trend confirmation) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence paths for trade frequency)
        if bull_cycle or above_sma200:
            # Path 1: Strong trend + pullback
            if adx_trending and kama_bullish and rsi_oversold and bull_1d:
                new_signal = LONG_SIZE
            # Path 2: KAMA crossover + RSI confirmation
            elif kama_bullish and rsi_14[i] < 50.0 and bull_1d:
                new_signal = LONG_SIZE * 0.8
            # Path 3: Simple oversold in bull cycle
            elif bull_cycle and rsi_oversold and adx_14[i] > 15.0:
                new_signal = LONG_SIZE * 0.7
            # Path 4: KAMA bullish + DI confirmation
            elif kama_bullish and plus_di_above and rsi_14[i] < 55.0:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (multiple confluence paths)
        if bear_cycle or below_sma200:
            # Path 1: Strong trend + bounce
            if adx_trending and kama_bearish and rsi_overbought and bear_1d:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 2: KAMA crossover + RSI confirmation
            elif kama_bearish and rsi_14[i] > 50.0 and bear_1d:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Path 3: Simple overbought in bear cycle
            elif bear_cycle and rsi_overbought and adx_14[i] > 15.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
            # Path 4: KAMA bearish + DI confirmation
            elif kama_bearish and minus_di_above and rsi_14[i] > 45.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h KAMA flip)
        if in_position and position_side > 0 and kama_bearish and adx_14[i] > 25.0:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish and adx_14[i] > 25.0:
            new_signal = 0.0
        
        # Major cycle flip exit (1w KAMA)
        if in_position and position_side > 0 and bear_cycle:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_cycle:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals