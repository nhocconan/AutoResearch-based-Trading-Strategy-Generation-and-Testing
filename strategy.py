#!/usr/bin/env python3
"""
Experiment #006: 12h Vol-Spike Mean Reversion with 1d Trend Bias

Hypothesis: Previous trend-following strategies failed because BTC/ETH spend 60%+
time in range/chop markets. This strategy flips the logic:
1. Wait for volatility SPIKE (ATR(7)/ATR(30) > 1.8) - signals panic/exhaustion
2. Wait for price at Bollinger extreme (2.5 std dev) - oversold/overbought
3. 1d HMA(21) determines BIAS only (long bias if price > 1d HMA, short if below)
4. Enter MEAN REVERSION trade (counter to recent move, WITH daily bias)

Why this should work:
- Vol spike + BB extreme = capitulation event (high prob reversal)
- 1d filter prevents catching falling knives in strong downtrends
- 12h TF = 20-40 trades/year (fee drag manageable)
- Works in both bull AND bear markets (mean reversion universal)

Key differences from failed experiments:
- NOT trend-following (those failed on BTC/ETH 2022-2025)
- NOT dual-regime switch (too complex, whipsawed)
- Simple vol-spike + mean reversion (proven in literature)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 3.0 * ATR(14) trailing
Target: 25-40 trades/year, Sharpe > 0.5 on all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_spike_mean_reversion_1d_bias_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std deviation."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std * std_mult)
    lower = sma - (std * std_mult)
    
    return upper.values, lower.values, sma.values

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
    rsi = np.where(np.isinf(rs), 100.0, rsi)
    rsi = np.where(np.isnan(rsi), 50.0, rsi)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.5)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility spike ratio (ATR short / ATR long)
    vol_spike_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0 and not np.isnan(atr_30[i]):
            vol_spike_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_spike_ratio[i] = 1.0
    
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
    consecutive_losses = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS (determines direction bias only) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30)
        
        # === BOLLINGER BAND EXTREMES ===
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        # === RSI EXTREMES (confirm exhaustion) ===
        rsi_oversold = rsi_14[i] < 25
        rsi_overbought = rsi_14[i] > 75
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # Reduce size after consecutive losses
        if consecutive_losses >= 2:
            current_size = current_size * 0.7
        
        # === ENTRY LOGIC (Mean Reversion with Daily Bias) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Vol spike + BB lower + RSI oversold + daily bias NOT strongly bearish
        long_condition = (
            vol_spike and
            at_bb_lower and
            rsi_oversold and
            (daily_bullish or not daily_bearish)  # Allow if neutral or bullish
        )
        
        # SHORT ENTRY: Vol spike + BB upper + RSI overbought + daily bias NOT strongly bullish
        short_condition = (
            vol_spike and
            at_bb_upper and
            rsi_overbought and
            (daily_bearish or not daily_bullish)  # Allow if neutral or bearish
        )
        
        if long_condition:
            new_signal = current_size
        
        if short_condition:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 days on 12h), allow weaker entries
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            # Weaker condition: just BB extreme + RSI (no vol spike required)
            if at_bb_lower and rsi_14[i] < 30 and daily_bullish:
                new_signal = current_size * 0.6
            elif at_bb_upper and rsi_14[i] > 70 and daily_bearish:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
                    consecutive_losses += 1
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
                    consecutive_losses += 1
        
        # === MEAN REVERSION EXIT (profit target) ===
        profit_exit = False
        if in_position and position_side != 0:
            if position_side > 0:
                # Exit long when price returns to BB middle
                if close[i] >= bb_mid[i] and (close[i] - entry_price) > 0.5 * atr_14[i]:
                    profit_exit = True
                    consecutive_losses = 0
            
            if position_side < 0:
                # Exit short when price returns to BB middle
                if close[i] <= bb_mid[i] and (entry_price - close[i]) > 0.5 * atr_14[i]:
                    profit_exit = True
                    consecutive_losses = 0
        
        # === DAILY BIAS REVERSAL EXIT ===
        bias_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns strongly bearish
            if position_side > 0 and daily_bearish and close[i] < hma_1d_21_aligned[i] * 0.98:
                bias_reversal = True
            # Exit short if 1d trend turns strongly bullish
            if position_side < 0 and daily_bullish and close[i] > hma_1d_21_aligned[i] * 1.02:
                bias_reversal = True
        
        # Apply stoploss, profit exit, or bias reversal
        if stoploss_triggered or profit_exit or bias_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals