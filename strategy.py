#!/usr/bin/env python3
"""
Experiment #024: 4h KAMA Trend + 12h/1d Bias with RSI Pullback

Hypothesis: Previous 4h strategies failed due to overly complex regime detection
and restrictive entry conditions. This strategy simplifies to proven patterns:
1. 1d HMA(21) for major trend bias (prevents counter-trend trades)
2. 12h KAMA(14) for adaptive trend direction (better than HMA in ranges)
3. 4h RSI(14) pullback entries in 35-65 range (not extreme - ensures trade frequency)
4. ATR(14) trailing stoploss at 2.5x for risk management
5. Discrete position sizing (0.25-0.30) to minimize fee churn

Why this should work:
- KAMA adapts to volatility better than HMA (Kaufman's Adaptive Moving Average)
- 12h trend filter reduces whipsaws vs pure 4h strategies
- RSI 35-65 range ensures enough trades (20-50/year target)
- 1d bias prevents major counter-trend losses (learned from 2022 crash)
- Based on current best baseline (mtf_4h_hma_rsi_pullback_1d_bias_v1) but with KAMA

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_12h1d_bias_v1"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = Change / Sum of absolute changes (Efficiency Ratio)
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = close_s.diff(period).abs()
    sum_changes = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / sum_changes
    er = er.fillna(0).clip(0, 1)
    
    # Calculate smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA iteratively
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]  # Initialize with price
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Calculate 12h indicators
    kama_12h_14 = calculate_kama(df_12h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    kama_12h_14_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_14)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_14 = calculate_kama(close, 14)
    kama_4h_30 = calculate_kama(close, 30)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_12h_14_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_4h_14[i]) or np.isnan(kama_4h_30[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H KAMA TREND ===
        # KAMA slope direction (compare to 3 bars ago for stability)
        kama_12h_slope_bullish = kama_12h_14_aligned[i] > kama_12h_14_aligned[i-3] if i > 3 else False
        kama_12h_slope_bearish = kama_12h_14_aligned[i] < kama_12h_14_aligned[i-3] if i > 3 else False
        
        # === 4H KAMA TREND ===
        kama_4h_bullish = kama_4h_14[i] > kama_4h_30[i]
        kama_4h_bearish = kama_4h_14[i] < kama_4h_30[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (RSI PULLBACK IN TREND) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 12h KAMA bullish + 1d bias bullish + 4h RSI pullback
        if kama_12h_slope_bullish and daily_bullish and kama_4h_bullish:
            # RSI pullback to 35-55 range (cooling off in uptrend)
            if 35 <= rsi_14[i] <= 55:
                new_signal = current_size
            # RSI bouncing from oversold with trend confirmation
            elif rsi_14[i] < 40 and rsi_14[i] > rsi_14[i-1]:
                new_signal = current_size
        
        # SHORT: 12h KAMA bearish + 1d bias bearish + 4h RSI pullback
        elif kama_12h_slope_bearish and daily_bearish and kama_4h_bearish:
            # RSI pullback to 45-65 range (cooling off in downtrend)
            if 45 <= rsi_14[i] <= 65:
                new_signal = -current_size
            # RSI bouncing from overbought with trend confirmation
            elif rsi_14[i] > 60 and rsi_14[i] < rsi_14[i-1]:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~5 days on 4h), allow weaker entry
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if kama_4h_bullish and daily_bullish and 40 <= rsi_14[i] <= 60:
                new_signal = current_size * 0.5
            elif kama_4h_bearish and daily_bearish and 40 <= rsi_14[i] <= 60:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_4h_bearish:
                trend_reversal = True
            if position_side < 0 and kama_4h_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals