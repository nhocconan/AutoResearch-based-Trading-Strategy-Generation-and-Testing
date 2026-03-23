#!/usr/bin/env python3
"""
Experiment #831: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: After 568+ failed strategies, the winning pattern is 4h timeframe with
1d HTF bias. 4h provides optimal trade frequency (20-50/year) with enough signal
quality. Key insight: use regime detection to switch between mean-reversion (chop)
and trend-following (trend), ensuring trades in ALL market conditions.

Strategy design:
1. 4h Primary timeframe (proven best for BTC/ETH/SOL)
2. 1d HMA(21) for HTF trend bias (aligns with larger trend)
3. 4h HMA(16/48) crossover for trend direction
4. 4h RSI(14) pullback entries (RSI<45 long, RSI>55 short in trend)
5. 4h Choppiness Index(14) for regime detection
6. 4h ATR(14) for trailing stop (2.5x)
7. Dual regime: mean-revert when CHOP>55, trend-follow when CHOP<45
8. Relaxed entry thresholds to ensure >=10 trades/symbol on train

Why this works:
- 4h TF: proven Sharpe=0.612 baseline, optimal trade frequency
- Regime switch: captures both trending AND ranging markets (2025 is bear/range)
- RSI pullback: enters on dips in uptrend, rallies in downtrend (better than breakout)
- HTF 1d bias: filters against major trend (reduces whipsaw)
- Relaxed thresholds: ensures trades on ALL symbols (BTC, ETH, SOL must all profit)

Position sizing: 0.25 (reduced), 0.30 (full) — discrete levels to minimize fee churn
Stoploss: 2.5x ATR trailing stop (signal → 0 when hit)
Target: Sharpe > 0.612, trades >= 80 train (20/year), >= 12 test (3/year), ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_regime_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (mean revert), CHOP < 45 = trending (trend follow).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range — volatility measure for stops."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # HMA trend indicators
    hma_fast_4h = calculate_hma(close, 16)
    hma_slow_4h = calculate_hma(close, 48)
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    FULL_SIZE = 0.30
    REDUCED_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(hma_fast_4h[i]) or np.isnan(hma_slow_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (1d HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === LTF TREND (4h HMA16/48 crossover) ===
        hma_bullish = hma_fast_4h[i] > hma_slow_4h[i]
        hma_bearish = hma_fast_4h[i] < hma_slow_4h[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === RSI SIGNALS (Relaxed for trade frequency) ===
        rsi_oversold = rsi_4h[i] < 45
        rsi_overbought = rsi_4h[i] > 55
        rsi_extreme_oversold = rsi_4h[i] < 30
        rsi_extreme_overbought = rsi_4h[i] > 70
        rsi_neutral = 40 <= rsi_4h[i] <= 60
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + HTF neutral or bullish
            if rsi_oversold and (trend_1d_bullish or rsi_neutral):
                desired_signal = FULL_SIZE
            
            # Short: RSI overbought + HTF neutral or bearish
            if rsi_overbought and (trend_1d_bearish or rsi_neutral):
                desired_signal = -FULL_SIZE
            
            # Extreme RSI override (ensures trades on all symbols)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: HMA bullish + RSI pullback + HTF confirmation
            if hma_bullish and rsi_oversold:
                if trend_1d_bullish:
                    desired_signal = FULL_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            # Short: HMA bearish + RSI rally + HTF confirmation
            if hma_bearish and rsi_overbought:
                if trend_1d_bearish:
                    desired_signal = -FULL_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            
            # HMA crossover entry (stronger signal)
            if i > 0 and not np.isnan(hma_fast_4h[i-1]) and not np.isnan(hma_slow_4h[i-1]):
                hma_cross_up = hma_fast_4h[i-1] <= hma_slow_4h[i-1] and hma_bullish
                hma_cross_down = hma_fast_4h[i-1] >= hma_slow_4h[i-1] and hma_bearish
                
                if hma_cross_up and trend_1d_bullish:
                    desired_signal = FULL_SIZE if desired_signal == 0 else desired_signal
                
                if hma_cross_down and trend_1d_bearish:
                    desired_signal = -FULL_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Conservative ===
        else:
            # Require both LTF and HTF alignment
            if hma_bullish and trend_1d_bullish and rsi_oversold:
                desired_signal = REDUCED_SIZE
            
            if hma_bearish and trend_1d_bearish and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            
            # RSI extreme override
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA bullish or HTF bullish
                if (hma_bullish or trend_1d_bullish) and rsi_4h[i] < 65:
                    desired_signal = FULL_SIZE if position_side > 0 else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if HMA bearish or HTF bearish
                if (hma_bearish or trend_1d_bearish) and rsi_4h[i] > 35:
                    desired_signal = -FULL_SIZE if position_side < 0 else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HMA and HTF turn bearish
            if hma_bearish and trend_1d_bearish:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HMA and HTF turn bullish
            if hma_bullish and trend_1d_bullish:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_4h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = FULL_SIZE if desired_signal >= FULL_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -FULL_SIZE if desired_signal <= -FULL_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals