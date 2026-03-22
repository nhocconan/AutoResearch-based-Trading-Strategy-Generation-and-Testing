#!/usr/bin/env python3
"""
Experiment #595: 15m Volatility Spike Mean Reversion with MTF Regime Filter

Hypothesis: After 594 experiments, the key insight is that volatility spike mean reversion
works best on lower timeframes (15m) where panic/recovery cycles are more frequent.

Strategy Logic:
1. VOL SPIKE DETECTION: ATR(7)/ATR(30) > 2.0 signals extreme volatility (panic/euphoria)
2. REGIME FILTER: 4h HMA determines bias (long only in bull, short only in bear)
3. ENTRY CONFIRMATION: 1h RSI extreme (<25 or >75) + price outside BB(20, 2.5)
4. EXIT: Mean reversion to BB mid + trailing stop at 2.5*ATR
5. This generates MORE trades than 12h/1d strategies while controlling drawdown

Why this should beat #593 (Sharpe=-0.281):
- 15m captures more vol spike events than 12h (more trade opportunities)
- Mean reversion works better in 2022 crash and 2025 bear than breakouts
- 4h HMA bias prevents counter-trend trades (major failure mode of #583-#594)
- Wider BB (2.5 std) ensures only extreme moves trigger entries
- ATR ratio filter catches panic bottoms and euphoria tops

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing (wider for 15m noise)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_spike_meanrev_4h_hma_1h_rsi_bb_atr_v1"
timeframe = "15m"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with wider std for extreme moves."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = atr_short / np.where(atr_long > 0, atr_long, np.inf)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.5)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss (separate from signal)
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_mid[i]):
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0  # Extreme volatility
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === BOLLINGER BAND EXTREMES ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        # === RSI EXTREMES (more extreme for 15m noise) ===
        rsi_oversold = rsi_14[i] < 25
        rsi_overbought = rsi_14[i] > 75
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: Vol spike + price below BB + RSI oversold + 4h bullish or neutral
        if vol_spike and below_bb_lower and rsi_oversold:
            if bull_bias or (not bear_bias and atr_ratio[i] > 2.5):
                new_signal = SIZE
        
        # SHORT: Vol spike + price above BB + RSI overbought + 4h bearish or neutral
        if vol_spike and above_bb_upper and rsi_overbought:
            if bear_bias or (not bull_bias and atr_ratio[i] > 2.5):
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === MEAN REVERSION EXIT (take profit at BB mid) ===
        take_profit = False
        if in_position and position_side != 0:
            if position_side > 0 and close[i] > bb_mid[i]:
                # Long position reached mean - take profit
                take_profit = True
            if position_side < 0 and close[i] < bb_mid[i]:
                # Short position reached mean - take profit
                take_profit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish strongly
            if position_side > 0 and bear_bias and close[i] < hma_4h_aligned[i] * 0.98:
                trend_reversal = True
            # Exit short if 4h trend turns bullish strongly
            if position_side < 0 and bull_bias and close[i] > hma_4h_aligned[i] * 1.02:
                trend_reversal = True
        
        # Apply stoploss or take profit or trend reversal
        if stoploss_triggered or take_profit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals