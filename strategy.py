#!/usr/bin/env python3
"""
Experiment #388: 4h Multi-TF HMA Trend + RSI Pullback + BB Volatility Filter

Hypothesis: After analyzing 387 experiments, the key insight is that strategies need:
1. STRONG multi-timeframe trend alignment (1w + 1d HMA) to avoid whipsaws
2. LOOSE entry conditions (RSI 35-65, not extremes) to generate enough trades
3. Volatility filter (BB width) to avoid entering during compression/breakout uncertainty
4. Asymmetric sizing based on trend strength (more capital when 1w/1d aligned)

Why this should beat Sharpe=0.676 baseline:
- Previous failures had RSI thresholds too strict (only RSI<30 or >70)
- This uses RSI 35-65 for pullback entries = MORE TRADES
- 1w HMA provides stronger trend filter than just 1d
- BB width percentile avoids entering during volatility compression (false breakouts)
- Position sizing scales with trend conviction (0.20 weak, 0.30 strong alignment)

STRATEGY COMPONENTS:
1. 1w HMA(21): Primary trend bias (weekly closes are most significant)
2. 1d HMA(21): Secondary trend confirmation
3. 4h RSI(14): Pullback entry timing (35-65 range, not extremes)
4. 4h Bollinger Band Width: Volatility filter (avoid <20th percentile)
5. ATR(14) trailing stop: 2.5x for risk management
6. Position sizing: 0.20-0.30 discrete, scales with trend alignment

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_trend_rsi_pullback_bbvol_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper.values, lower.values, bandwidth.values

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate rolling percentile of BB width to detect compression/expansion."""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    for i in range(lookback, n):
        window = bandwidth[i-lookback+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current = bandwidth[i]
            percentile[i] = np.sum(valid <= current) / len(valid) * 100
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TF TREND ALIGNMENT ===
        # Strong bull: price > 1d HMA > 1w HMA
        strong_bull = (close[i] > hma_1d_aligned[i]) and (hma_1d_aligned[i] > hma_1w_aligned[i])
        # Strong bear: price < 1d HMA < 1w HMA
        strong_bear = (close[i] < hma_1d_aligned[i]) and (hma_1d_aligned[i] < hma_1w_aligned[i])
        # Weak bull: price > 1d HMA but 1d < 1w (transitioning)
        weak_bull = (close[i] > hma_1d_aligned[i]) and (hma_1d_aligned[i] <= hma_1w_aligned[i])
        # Weak bear: price < 1d HMA but 1d > 1w (transitioning)
        weak_bear = (close[i] < hma_1d_aligned[i]) and (hma_1d_aligned[i] >= hma_1w_aligned[i])
        
        # === VOLATILITY FILTER ===
        # Only trade when BB width > 20th percentile (avoid compression)
        vol_ok = bb_width_pct[i] > 20.0
        
        # === RSI PULLBACK ENTRY (LOOSE thresholds for more trades) ===
        # Long: RSI 35-50 in bull trend (pullback, not oversold extreme)
        rsi_long_pullback = 35.0 <= rsi[i] <= 50.0
        # Short: RSI 50-65 in bear trend (pullback, not overbought extreme)
        rsi_short_pullback = 50.0 <= rsi[i] <= 65.0
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        # Strong alignment = 0.30, weak alignment = 0.20
        size_strong = 0.30
        size_weak = 0.20
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if vol_ok:
            # LONG entries
            if strong_bull and rsi_long_pullback:
                new_signal = size_strong
            elif weak_bull and rsi_long_pullback:
                new_signal = size_weak
            
            # SHORT entries
            if strong_bear and rsi_short_pullback:
                new_signal = -size_strong
            elif weak_bear and rsi_short_pullback:
                new_signal = -size_weak
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if trend turns bearish
        if in_position and position_side > 0 and new_signal != 0.0:
            if strong_bear or weak_bear:
                new_signal = 0.0
        
        # Exit short if trend turns bullish
        if in_position and position_side < 0 and new_signal != 0.0:
            if strong_bull or weak_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals