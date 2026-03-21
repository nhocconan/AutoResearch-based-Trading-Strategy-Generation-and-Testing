#!/usr/bin/env python3
"""
Experiment #415: 15m Mean Reversion + 4h Trend Filter + Fisher Transform + RSI + ATR Stop
Hypothesis: 15m timeframe is ideal for mean-reversion strategies. Combining 4h HMA trend bias
with 15m RSI extremes and Fisher Transform reversals should generate frequent trades while
respecting higher-timeframe direction. Multiple entry paths ensure >=10 trades/symbol.
Key features: 4h HMA for trend bias, 15m RSI(7) for quick extremes, Fisher(9) for reversals,
Bollinger Bands for squeeze detection, Z-score for mean reversion confirmation.
Position size: 0.25 discrete, stoploss 2*ATR for 15m timeframe.
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_mr_4h_hma_fisher_rsi_bb_zscore_atr_v1"
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

def calculate_fisher(close, high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    smoothed_prev = 0.0
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            if i > period:
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (close[i] - lowest) / (highest - lowest) - 1.0
        normalized = np.clip(normalized, -0.99, 0.99)
        
        if i == period:
            smoothed = normalized
        else:
            smoothed = 0.67 * normalized + 0.33 * smoothed_prev
        
        smoothed_prev = smoothed
        fisher[i] = 0.5 * np.log((1.0 + smoothed) / (1.0 - smoothed))
        
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_rsi(close, period=7):
    """Calculate RSI with shorter period for 15m responsiveness."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, trigger = calculate_fisher(close, high, low, 9)
    rsi = calculate_rsi(close, 7)  # Shorter period for 15m
    sma_bb, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    sma50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_bb[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (higher timeframe direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # RSI extremes (mean reversion signals)
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        rsi_neutral_long = rsi[i] > 30 and rsi[i] < 60
        rsi_neutral_short = rsi[i] > 40 and rsi[i] < 70
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        fisher_bull_cross = fisher[i] > -1.2 and trigger[i] <= -1.2
        fisher_bear_cross = fisher[i] < 1.2 and trigger[i] >= 1.2
        fisher_turning_up = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_turning_down = fisher[i] < fisher[i-1] if i > 0 else False
        
        # Bollinger Band signals
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        bb_squeeze = (bb_upper[i] - bb_lower[i]) < 0.02 * sma_bb[i]  # Narrow bands
        
        # Z-score mean reversion
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # SMA50 trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: RSI oversold + 4h bullish + Fisher turning up (primary mean reversion)
        if rsi_oversold and trend_bullish and fisher_turning_up:
            new_signal = SIZE_ENTRY
        # Path 2: Fisher cross + 4h bullish + RSI neutral
        elif fisher_bull_cross and trend_bullish and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 3: Price below BB + 4h bullish + Z-score oversold
        elif price_below_bb and trend_bullish and zscore_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: Fisher oversold + turning up + above SMA50 (trend-following MR)
        elif fisher_oversold and fisher_turning_up and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 5: RSI oversold + Fisher cross (double confirmation)
        elif rsi_oversold and fisher_bull_cross:
            new_signal = SIZE_ENTRY
        # Path 6: 4h bullish + RSI > 40 + Fisher > -0.5 (momentum continuation)
        elif trend_bullish and rsi[i] > 40 and fisher[i] > -0.5 and above_sma50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: RSI overbought + 4h bearish + Fisher turning down (primary mean reversion)
        if rsi_overbought and trend_bearish and fisher_turning_down:
            new_signal = -SIZE_ENTRY
        # Path 2: Fisher cross + 4h bearish + RSI neutral
        elif fisher_bear_cross and trend_bearish and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Price above BB + 4h bearish + Z-score overbought
        elif price_above_bb and trend_bearish and zscore_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: Fisher overbought + turning down + below SMA50 (trend-following MR)
        elif fisher_overbought and fisher_turning_down and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 5: RSI overbought + Fisher cross (double confirmation)
        elif rsi_overbought and fisher_bear_cross:
            new_signal = -SIZE_ENTRY
        # Path 6: 4h bearish + RSI < 60 + Fisher < 0.5 (momentum continuation)
        elif trend_bearish and rsi[i] < 60 and fisher[i] < 0.5 and below_sma50:
            new_signal = -SIZE_ENTRY
        
        # Override: if strong opposite signal, flip position
        if new_signal == 0.0 and position_side > 0:
            # Check for strong short signal
            if rsi_overbought and fisher_bear_cross:
                new_signal = -SIZE_ENTRY
        elif new_signal == 0.0 and position_side < 0:
            # Check for strong long signal
            if rsi_oversold and fisher_bull_cross:
                new_signal = SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals