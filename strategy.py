#!/usr/bin/env python3
"""
Experiment #239: 12h Volatility Spike Mean Reversion with 1d HMA Trend Filter

Hypothesis: On 12h timeframe, volatility spikes followed by mean reversion provide 
high-probability entries. When ATR(7)/ATR(30) > 1.5 (vol spike) + price at BB(20,2.0) 
extremes + RSI(14) extreme (30/70), expect vol crush and price reversion to mean.
1d HMA provides higher-timeframe trend bias to avoid counter-trend trades.

Why 12h might work:
- 12h has fewer false signals than lower TFs (15m-1h failed badly in exp #229-237)
- Previous 12h attempts (#227, #233) had near-zero Sharpe but not catastrophic losses
- Vol spike reversion works in both bull and bear markets (unlike pure trend)
- 12h captures multi-day swings without noise of lower TFs
- Conservative sizing (0.28) controls drawdown during 2022 crash

Key improvements over failed experiments:
- #236 (30m Fisher): too noisy, wrong TF → use 12h for cleaner signals
- #235 (15m trend): whipsaw in range → use mean reversion, not trend
- #232 (4h KAMA): no vol filter → add ATR ratio for vol spike detection
- All failed strategies: no regime adaptation → use BB width + ATR ratio

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels (conservative for 12h swing trades)
Stoploss: 2.5 * ATR(14) trailing
Target: 20-40 trades/year on 12h (enough for Sharpe calc, low fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_spike_bb_rsi_1d_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Normalized bandwidth
    
    return upper.values, lower.values, sma.values, width.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # Volatility spike ratio: ATR(7) / ATR(30)
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    zscore_20 = calculate_zscore(close, 20)
    
    # Bollinger Band width percentile (for regime detection)
    bb_width_pct = pd.Series(bb_width).rolling(window=60, min_periods=60).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if x.max() > x.min() else 0.5
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    SIZE_HALF = 0.14
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_idx = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        # vol_ratio > 1.5 = volatility spike (mean reversion opportunity)
        # vol_ratio < 1.0 = low vol (trend may develop)
        vol_spike = vol_ratio[i] > 1.5
        vol_low = vol_ratio[i] < 1.0
        
        # === BB REGIME ===
        # bb_width_pct < 0.2 = BB squeeze (breakout likely)
        # bb_width_pct > 0.8 = BB expansion (mean reversion likely)
        bb_expansion = bb_width_pct[i] > 0.7 if not np.isnan(bb_width_pct[i]) else False
        bb_squeeze = bb_width_pct[i] < 0.3 if not np.isnan(bb_width_pct[i]) else False
        
        # === MEAN REVERSION SIGNALS ===
        # Long: price at BB lower + RSI < 35 + vol spike OR bb expansion
        price_at_bb_lower = close[i] <= bb_lower[i]
        price_at_bb_upper = close[i] >= bb_upper[i]
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        zscore_extreme_low = zscore_20[i] < -1.5
        zscore_extreme_high = zscore_20[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Require: price at BB lower + RSI oversold + (vol spike OR bb expansion)
        # 1d trend: bullish or neutral (not strongly bearish)
        if price_at_bb_lower and rsi_oversold and (vol_spike or bb_expansion):
            if bull_trend_1d or zscore_20[i] > -1.0:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Require: price at BB upper + RSI overbought + (vol spike OR bb expansion)
        # 1d trend: bearish or neutral (not strongly bullish)
        if price_at_bb_upper and rsi_overbought and (vol_spike or bb_expansion):
            if bear_trend_1d or zscore_20[i] < 1.0:
                new_signal = -SIZE_BASE
        
        # === BREAKOUT SIGNALS (BB Squeeze) ===
        # When BB squeezes, breakout in direction of 1d trend
        if bb_squeeze and not in_position:
            if bull_trend_1d and close[i] > bb_mid[i] and rsi_14[i] > 50:
                new_signal = SIZE_BASE * 0.7  # Smaller size for breakout
            if bear_trend_1d and close[i] < bb_mid[i] and rsi_14[i] < 50:
                new_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr_14[entry_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr_14[entry_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === EXIT ON VOL NORMALIZATION ===
        # If in mean reversion trade and vol_ratio drops below 1.1, exit
        if in_position and vol_ratio[i] < 1.1 and abs(new_signal) < SIZE_BASE:
            # Vol crush complete, close position
            if position_side > 0 and close[i] > entry_price:
                new_signal = 0.0
            if position_side < 0 and close[i] < entry_price:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position and signals[i-1] != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals