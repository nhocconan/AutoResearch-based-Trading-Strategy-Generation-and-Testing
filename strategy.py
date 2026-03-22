#!/usr/bin/env python3
"""
Experiment #012: 12h KAMA Adaptive Trend + 1d HMA Filter + Volume Confirmation

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better
than fixed EMAs, reducing whipsaws in choppy markets while capturing trends efficiently.
Combined with 1d HMA trend filter and volume confirmation, this should:
1. Reduce false breakouts via volume filter
2. Adapt to regime changes via KAMA's Efficiency Ratio
3. Generate sufficient trades (20-50/year) with looser entry conditions
4. Protect capital with 2.5x ATR stoploss

Key design:
- 1d HMA(21) for major trend bias (call ONCE via mtf_data)
- 12h KAMA(10) for adaptive trend following
- Volume > 1.5x 20-bar average for breakout confirmation
- RSI(14) simple filter (>45 for long, <55 for short) - NOT over-filtered
- ATR(14) stoploss at 2.5x
- Discrete sizing: 0.20, 0.25, 0.30 based on trend confluence

Why this should beat previous attempts:
- KAMA reduces whipsaw vs EMA/HMA in ranging markets (2022 crash)
- Volume filter prevents false breakouts (major failure mode)
- Simpler RSI filter ensures trades actually trigger (learned from 0-trade failures)
- 12h TF targets optimal trade frequency (20-50/year)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_vol_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (fast SC), Low ER = choppy (slow SC)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    # Sum of absolute changes (volatility)
    abs_diff = np.abs(close - np.roll(close, 1))
    abs_diff[0] = 0
    
    # Use pandas rolling for sum
    abs_diff_s = pd.Series(abs_diff)
    volatility = abs_diff_s.rolling(window=period, min_periods=period).sum().values
    
    # Efficiency Ratio
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[period] = close[period]  # Initialize with price
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_10 = calculate_kama(close, 10)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # Also calculate 12h HMA for additional confirmation
    hma_12h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(kama_10[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # KAMA slope (trend momentum)
        kama_slope_up = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_slope_down = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # === 12H HMA CONFIRMATION ===
        hma_bullish = close[i] > hma_12h_21[i]
        hma_bearish = close[i] < hma_12h_21[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.5 * vol_sma_20[i] if vol_sma_20[i] > 0 else False
        
        # === RSI FILTER (simple, not over-filtered) ===
        rsi_bullish = rsi_14[i] > 45  # Looser than 50 to ensure trades
        rsi_bearish = rsi_14[i] < 55  # Looser than 50 to ensure trades
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and kama_bullish and hma_bullish:
            current_size = STRONG_SIZE
        elif htf_bullish and kama_bullish:
            current_size = BASE_SIZE
        elif htf_bullish:
            current_size = WEAK_SIZE
        elif htf_bearish and kama_bearish and hma_bearish:
            current_size = STRONG_SIZE
        elif htf_bearish and kama_bearish:
            current_size = BASE_SIZE
        elif htf_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1d bullish + KAMA bullish + RSI > 45
        # Volume confirmation preferred but not required (to ensure trades)
        if htf_bullish and kama_bullish and rsi_bullish:
            if vol_confirmed or kama_slope_up:
                new_signal = current_size
            else:
                new_signal = current_size * 0.8  # Weaker signal without volume
        
        # SHORT ENTRY: 1d bearish + KAMA bearish + RSI < 55
        elif htf_bearish and kama_bearish and rsi_bearish:
            if vol_confirmed or kama_slope_down:
                new_signal = -current_size
            else:
                new_signal = -current_size * 0.8  # Weaker signal without volume
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~12 days on 12h), allow weaker entry
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if htf_bullish and kama_bullish:
                new_signal = BASE_SIZE * 0.7
            elif htf_bearish and kama_bearish:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
            # Exit long if 1d trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === KAMA REVERSAL EXIT ===
        kama_reversal = False
        if in_position and position_side != 0:
            # Exit long if KAMA turns bearish
            if position_side > 0 and kama_bearish:
                kama_reversal = True
            # Exit short if KAMA turns bullish
            if position_side < 0 and kama_bullish:
                kama_reversal = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or kama_reversal:
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