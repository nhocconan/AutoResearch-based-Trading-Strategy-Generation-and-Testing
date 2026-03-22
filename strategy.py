#!/usr/bin/env python3
"""
Experiment #007: 15m RSI Mean Reversion + 1h HMA Trend Bias + BB Confirmation
Hypothesis: 15m timeframe needs faster signals with looser entry conditions to generate trades.
RSI(14) extremes (<30/>70) combined with Bollinger Band position provide frequent mean reversion entries.
1h HMA provides trend BIAS (boosts size when aligned) but doesn't block entries - critical for trade generation.
ATR trailing stop at 2.5*ATR limits drawdown. Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn.
Key insight from failures: entry conditions were TOO STRICT causing 0 trades. This strategy loosens filters.
Timeframe: 15m (REQUIRED), HTF: 1h via mtf_data helper.
Position sizing: 0.25 base, 0.30 when HTF trend aligned, stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_bb_1h_hma_bias_v1"
timeframe = "15m"
leverage = 1.0

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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    # Position within bands: 0=lower, 0.5=middle, 1=upper
    bb_position = (close - lower) / (upper - lower + 1e-10)
    bb_position = np.clip(bb_position, 0, 1)
    bb_position = np.nan_to_num(bb_position, nan=0.5)
    return upper, lower, bandwidth, sma, bb_position

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = np.nan_to_num(atr, nan=0.0)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma, bb_position = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Additional trend filter (loose)
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50 = np.nan_to_num(ema_50, nan=close)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if atr[i] == 0 or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            # HTF not ready, can still trade with base size
            hma_1h_valid = False
        else:
            hma_1h_valid = True
        
        # 1h trend bias (HTF) - boosts size but doesn't block entry
        if hma_1h_valid:
            bull_trend = close[i] > hma_1h_aligned[i]
            bear_trend = close[i] < hma_1h_aligned[i]
        else:
            bull_trend = False
            bear_trend = False
        
        # RSI signals (mean reversion) - LOOSE thresholds for trade generation
        rsi_oversold = rsi[i] < 35  # Looser than 30
        rsi_overbought = rsi[i] > 65  # Looser than 70
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        # Price position vs Bollinger Bands
        price_near_lower = bb_position[i] < 0.15  # Bottom 15% of BB range
        price_near_upper = bb_position[i] > 0.85  # Top 15% of BB range
        price_below_lower = close[i] < bb_lower[i]  # Actually below lower band
        price_above_upper = close[i] > bb_upper[i]  # Actually above upper band
        
        # EMA trend (loose filter)
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY === (multiple paths to ensure trades generate)
        # Path 1: RSI extreme oversold + price near lower BB (strongest signal)
        if rsi_extreme_oversold and (price_near_lower or price_below_lower):
            new_signal = SIZE_MAX if bull_trend else SIZE_BASE
        # Path 2: RSI oversold + price below lower BB
        elif rsi_oversold and price_below_lower:
            new_signal = SIZE_BASE
        # Path 3: RSI oversold + price near lower BB + EMA bullish
        elif rsi_oversold and price_near_lower and ema_bullish:
            new_signal = SIZE_BASE
        # Path 4: RSI very oversold (any BB position) - catch deep dips
        elif rsi[i] < 20:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY === (multiple paths to ensure trades generate)
        # Path 1: RSI extreme overbought + price near upper BB (strongest signal)
        if rsi_extreme_overbought and (price_near_upper or price_above_upper):
            new_signal = -SIZE_MAX if bear_trend else -SIZE_BASE
        # Path 2: RSI overbought + price above upper BB
        elif rsi_overbought and price_above_upper:
            new_signal = -SIZE_BASE
        # Path 3: RSI overbought + price near upper BB + EMA bearish
        elif rsi_overbought and price_near_upper and ema_bearish:
            new_signal = -SIZE_BASE
        # Path 4: RSI very overbought (any BB position) - catch rallies
        elif rsi[i] > 80:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals