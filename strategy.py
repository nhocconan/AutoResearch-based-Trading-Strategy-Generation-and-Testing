#!/usr/bin/env python3
"""
Experiment #532: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX Regime + RSI Pullback

Hypothesis: After analyzing 477 failed strategies, the pattern shows:
- Pure HMA/EMA crossovers fail (whipsaw in 2022 crash)
- Choppiness Index strategies consistently negative Sharpe
- Volatility spike strategies mostly failed
- KAMA + ADX + trend following showed promise (#529 Sharpe=0.169, Return=+28.6%)

This strategy uses:
1. KAMA (Kaufman Adaptive Moving Average) on 12h - adapts to volatility, reduces whipsaw
2. ADX(14) regime filter - only trade when ADX > 20 (trending market)
3. 1d KAMA(21) for major trend bias - align with HTF direction
4. 1w KAMA(50) for secular trend - avoid counter-secular trades
5. RSI(14) pullback entries - enter on retracements in trending markets
6. ATR(14) 2.5x trailing stop for risk management
7. Asymmetric sizing - larger positions in strong trends (ADX > 30)

Why this might work:
- KAMA adapts to market noise (ER-based), unlike fixed EMA/HMA
- ADX filter avoids range-bound whipsaw (major failure mode in 2022)
- 1d/1w HTF alignment prevents counter-trend trades
- RSI pullback entries avoid chasing breakouts
- 12h TF targets 25-40 trades/year (optimal fee/trade ratio per Rule 10)

Position sizing: 0.25 base, 0.35 when ADX > 30 (strong trend)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_rsi_pullback_1d1w_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |change| / sum(|individual changes|)
    High ER = trending (fast SC), Low ER = ranging (slow SC)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    # Sum of absolute individual changes (volatility/noise)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    volatility[:period] = np.nan
    
    # Efficiency Ratio (ER)
    er = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    er[:period] = np.nan
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] ** 2 * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF KAMA for major trend
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_50 = calculate_kama(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF KAMA for secular trend
    kama_1w_50 = calculate_kama(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    kama_1d_50_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_50)
    kama_1w_50_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # KAMA for 12h entries (adaptive to volatility)
    kama_12h_10 = calculate_kama(close, period=10)
    kama_12h_30 = calculate_kama(close, period=30)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(adx_14[i]):
            continue
        if np.isnan(kama_1d_21_aligned[i]) or np.isnan(kama_1w_50_aligned[i]):
            continue
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (1d and 1w) ===
        # Bullish: price > 1d KAMA(21) AND 1d KAMA(21) > 1d KAMA(50)
        bull_1d = close[i] > kama_1d_21_aligned[i]
        bull_1d_slope = kama_1d_21_aligned[i] > kama_1d_50_aligned[i]
        
        # Bearish: price < 1d KAMA(21) AND 1d KAMA(21) < 1d KAMA(50)
        bear_1d = close[i] < kama_1d_21_aligned[i]
        bear_1d_slope = kama_1d_21_aligned[i] < kama_1d_50_aligned[i]
        
        # Secular trend (1w) - avoid counter-secular trades
        bull_1w = close[i] > kama_1w_50_aligned[i]
        bear_1w = close[i] < kama_1w_50_aligned[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending market (trade), ADX < 20 = ranging (skip or mean revert)
        trending_regime = adx_14[i] > 20.0
        strong_trend = adx_14[i] > 30.0
        
        # DI crossover for direction confirmation
        di_bull = plus_di[i] > minus_di[i]
        di_bear = plus_di[i] < minus_di[i]
        
        # === 12H KAMA CROSSOVER ===
        # Fast KAMA crosses slow KAMA
        kama_cross_up = (kama_12h_10[i] > kama_12h_30[i]) and (kama_12h_10[i-1] <= kama_12h_30[i-1])
        kama_cross_down = (kama_12h_10[i] < kama_12h_30[i]) and (kama_12h_10[i-1] >= kama_12h_30[i-1])
        
        # KAMA alignment
        kama_aligned_bull = kama_12h_10[i] > kama_12h_30[i]
        kama_aligned_bear = kama_12h_10[i] < kama_12h_30[i]
        
        # === RSI PULLBACK ENTRIES ===
        # Long: RSI pulls back to 40-50 in uptrend
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        # Short: RSI bounces to 50-60 in downtrend
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        
        # Extreme RSI for mean reversion in ranges
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === POSITION SIZING ===
        current_size = STRONG_SIZE if strong_trend else BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for frequency)
        # Condition 1: KAMA crossover up + 1d bull + trending + RSI pullback
        if kama_cross_up and bull_1d and trending_regime and rsi_pullback_long:
            new_signal = current_size
        # Condition 2: KAMA aligned bull + 1d bull + DI bull + RSI not overbought
        elif kama_aligned_bull and bull_1d and di_bull and rsi_14[i] < 70.0:
            new_signal = BASE_SIZE
        # Condition 3: Strong trend + KAMA crossover + 1w bull (secular alignment)
        elif strong_trend and kama_cross_up and bull_1w:
            new_signal = STRONG_SIZE
        # Condition 4: RSI oversold + 1d bull (pullback in uptrend)
        elif rsi_oversold and bull_1d and bull_1d_slope:
            new_signal = BASE_SIZE
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: KAMA crossover down + 1d bear + trending + RSI pullback
            if kama_cross_down and bear_1d and trending_regime and rsi_pullback_short:
                new_signal = -current_size
            # Condition 2: KAMA aligned bear + 1d bear + DI bear + RSI not oversold
            elif kama_aligned_bear and bear_1d and di_bear and rsi_14[i] > 30.0:
                new_signal = -BASE_SIZE
            # Condition 3: Strong trend + KAMA crossover + 1w bear (secular alignment)
            elif strong_trend and kama_cross_down and bear_1w:
                new_signal = -STRONG_SIZE
            # Condition 4: RSI overbought + 1d bear (bounce in downtrend)
            elif rsi_overbought and bear_1d and bear_1d_slope:
                new_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or extreme RSI) ===
        # Exit long on regime flip to bear
        if in_position and position_side > 0:
            if bear_1d and bear_1d_slope:
                new_signal = 0.0
            elif rsi_14[i] > 75.0:  # Extreme overbought - take profit
                new_signal = 0.0
            elif adx_14[i] < 15.0:  # Trend exhausted
                new_signal = 0.0
        
        # Exit short on regime flip to bull
        if in_position and position_side < 0:
            if bull_1d and bull_1d_slope:
                new_signal = 0.0
            elif rsi_14[i] < 25.0:  # Extreme oversold - take profit
                new_signal = 0.0
            elif adx_14[i] < 15.0:  # Trend exhausted
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals