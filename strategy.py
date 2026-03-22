#!/usr/bin/env python3
"""
Experiment #002: 30m Multi-Timeframe Fisher-RSI Mean Reversion with 4h Trend Bias
Hypothesis: 30m captures intraday swings while 4h HMA provides trend context.
Fisher Transform (period=9) catches reversals in bear/range markets better than EMA.
RSI(3) extreme readings + Fisher crosses generate frequent entries (>=10 trades/symbol).
Volatility filter (ATR ratio) avoids low-vol whipsaws. Conservative sizing (0.25-0.30)
with 2.5*ATR stoploss. Designed to work on BTC/ETH through 2022 crash and 2025 bear.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_rsi_vol_reversion_4h_hma_v1"
timeframe = "30m"
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
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        # Calculate price position within range
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val == 0:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price position
        x = (hl2[-1] - lowest) / range_val
        x = 0.999 * x + 0.001  # Clamp to avoid infinity
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_val
        else:
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
    
    return fisher

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    pct = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, sma, width, pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_3 = calculate_rsi(close, 3)  # Fast RSI for Connors-style
    fisher = calculate_fisher_transform(high, low, 9)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_width, bb_pct = calculate_bollinger(close, 20, 2.0)
    
    # ATR ratio for vol filter (ATR7/ATR30)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_above_avg = volume > vol_ma
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(rsi_3[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - relaxed for more trades
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Volatility filter - avoid ultra-low vol (whipsaw prone)
        vol_ok = atr_ratio[i] > 0.5  # Not in extreme vol crush
        
        # Fisher Transform signals (proven in bear markets)
        fisher_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # RSI(3) extreme readings (Connors-style mean reversion)
        rsi3_oversold = rsi_3[i] < 15
        rsi3_overbought = rsi_3[i] > 85
        rsi3_turning_up = rsi_3[i] > rsi_3[i-1] if i > 0 else False
        rsi3_turning_down = rsi_3[i] < rsi_3[i-1] if i > 0 else False
        
        # RSI(14) zones
        rsi14_oversold = rsi_14[i] < 35
        rsi14_overbought = rsi_14[i] > 65
        
        # Bollinger positions
        bb_near_lower = bb_pct[i] < 0.2
        bb_near_upper = bb_pct[i] > 0.8
        bb_below_lower = close[i] < bb_lower[i]
        bb_above_upper = close[i] > bb_upper[i]
        
        # ADX trend strength
        adx_weak = adx[i] < 25  # Range market
        adx_moderate = adx[i] > 18
        
        # Volume confirmation
        vol_confirmed = vol_above_avg[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 4h bullish + Fisher long cross + RSI3 oversold
        if hma_4h_bullish and fisher_long and rsi3_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h bullish + BB below lower + RSI14 oversold (overshoot)
        elif hma_4h_bullish and bb_below_lower and rsi14_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: Fisher long + RSI3 turning up + volume (any 4h bias)
        elif fisher_long and rsi3_turning_up and vol_confirmed:
            new_signal = SIZE_ENTRY
        
        # Path 4: BB near lower + RSI3 oversold + turning up (mean reversion)
        elif bb_near_lower and rsi3_oversold and rsi3_turning_up:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h bullish + ADX weak (range) + RSI14 oversold
        elif hma_4h_bullish and adx_weak and rsi14_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 6: RSI3 extreme oversold (<10) + any volume (catch panic bottoms)
        elif rsi_3[i] < 10 and vol_ok:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 4h bearish + Fisher short cross + RSI3 overbought
        if hma_4h_bearish and fisher_short and rsi3_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h bearish + BB above upper + RSI14 overbought (overshoot)
        elif hma_4h_bearish and bb_above_upper and rsi14_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Fisher short + RSI3 turning down + volume (any 4h bias)
        elif fisher_short and rsi3_turning_down and vol_confirmed:
            new_signal = -SIZE_ENTRY
        
        # Path 4: BB near upper + RSI3 overbought + turning down (mean reversion)
        elif bb_near_upper and rsi3_overbought and rsi3_turning_down:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h bearish + ADX weak (range) + RSI14 overbought
        elif hma_4h_bearish and adx_weak and rsi14_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 6: RSI3 extreme overbought (>90) + any volume (catch panic tops)
        elif rsi_3[i] > 90 and vol_ok:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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