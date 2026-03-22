#!/usr/bin/env python3
"""
Experiment #011: 4h HMA Trend + RSI Pullback with 1d/1w Bias

Hypothesis: Previous regime-switching strategies failed due to over-complexity.
This strategy uses a simpler, proven approach:
1. 4h HMA crossover for primary trend signal
2. 1d HMA for major trend bias (only trade with daily trend)
3. 1w HMA for secular trend filter (avoid counter-trend trades)
4. RSI(14) pullback entry (40-60 range, not extremes)
5. ATR-based stoploss (2.5 ATR)

Why this should work:
- 4h timeframe proven to work (current best is 4h-based)
- Triple HMA alignment reduces whipsaw in bear markets
- RSI pullback (not extreme) generates more trades than Connors extremes
- Simpler logic = fewer conflicting filters = more trade frequency
- Discrete position sizing (0.25-0.30) minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
Target: 20-50 trades/year, Sharpe > 0.028 (beat current best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d1w_bias_v2"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_50[i]):
            continue
        
        # === 1W SECULAR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i] and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i] and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        daily_neutral = not daily_bullish and not daily_bearish
        
        # === 4H HMA TREND ===
        hma_4h_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_4h_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 40-50 in uptrend
        rsi_long_pullback = 38 < rsi_14[i] < 52
        # Short: RSI pulled back to 50-62 in downtrend
        rsi_short_pullback = 48 < rsi_14[i] < 62
        
        # === VOLATILITY FILTER ===
        # Only trade when ATR is not at extreme (avoid panic entries)
        atr_median = np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else atr_14[i]
        atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
        vol_normal = 0.5 < atr_ratio < 2.5
        
        # === POSITION SIZING ===
        # Reduce size in high volatility
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.round(current_size, 2)
        current_size = np.clip(current_size, 0.20, 0.32)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: All trend aligned + RSI pullback
        if weekly_bullish and daily_bullish and hma_4h_bullish:
            if rsi_long_pullback and vol_normal:
                if close[i] > sma_50[i]:  # Above 50 SMA confirmation
                    new_signal = current_size
        
        # SHORT ENTRY: All trend aligned + RSI pullback
        elif weekly_bearish and daily_bearish and hma_4h_bearish:
            if rsi_short_pullback and vol_normal:
                if close[i] < sma_50[i]:  # Below 50 SMA confirmation
                    new_signal = -current_size
        
        # === NEUTRAL MARKET: Reduced position, simpler logic ===
        if daily_neutral and not in_position:
            # Only 4h trend + RSI, skip daily/weekly filters
            if hma_4h_bullish and rsi_long_pullback and vol_normal:
                new_signal = current_size * 0.6
            elif hma_4h_bearish and rsi_short_pullback and vol_normal:
                new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~7 days on 4h), force entry with weaker signal
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and rsi_14[i] < 55:
                new_signal = current_size * 0.5
            elif hma_4h_bearish and rsi_14[i] > 45:
                new_signal = -current_size * 0.5
        
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
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            if position_side < 0 and hma_4h_bullish:
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