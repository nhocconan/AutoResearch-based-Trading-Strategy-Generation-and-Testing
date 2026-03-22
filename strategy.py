#!/usr/bin/env python3
"""
Experiment #583: 15m Multi-Regime Strategy with 4h HMA Trend Bias

Hypothesis: 15m timeframe needs strong HTF filter to avoid noise.
Using 4h HMA for trend bias + Choppiness Index for regime detection.
- Range regime (CHOP>61.8): Mean reversion at BB extremes
- Trend regime (CHOP<38.2): Pullback entries with trend
- ADX>18 filter ensures minimum trend strength
- 2*ATR stoploss protects against crashes
- Discrete sizing 0.25 to minimize fee churn

Why this should work:
1. 4h HMA provides stable trend bias (less noise than 1h)
2. Choppiness Index adapts to market regime
3. Different entry logic per regime avoids one-size-fits-all failure
4. 15m captures intraday moves while 4h filter prevents counter-trend
5. Conservative sizing (0.25) limits drawdown during 2022 crash
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_chop_4h_hma_bb_rsi_adaptive_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR calculation
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll).replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop_14[i] > 55.0  # Slightly lower threshold for more trades
        is_trend = chop_14[i] < 45.0  # Slightly higher threshold for more trades
        
        # === ADX FILTER (loose threshold for more trades) ===
        trend_confirmed = adx_14[i] > 18  # Lower than typical 25 for more signals
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        if is_range:
            # === RANGE REGIME: Mean Reversion at BB Extremes ===
            # Long: RSI oversold + price near lower BB + 4h bull bias preferred
            if rsi_14[i] < 35 and close[i] <= bb_lower[i] * 1.002:
                if bull_bias:
                    new_signal = SIZE
                elif adx_14[i] < 25:  # Weak trend, still enter
                    new_signal = SIZE * 0.6  # Reduced size in bear range
            
            # Short: RSI overbought + price near upper BB + 4h bear bias preferred
            if rsi_14[i] > 65 and close[i] >= bb_upper[i] * 0.998:
                if bear_bias:
                    new_signal = -SIZE
                elif adx_14[i] < 25:  # Weak trend, still enter
                    new_signal = -SIZE * 0.6  # Reduced size in bull range
        
        elif is_trend:
            # === TREND REGIME: Pullback Entries ===
            # Long: 4h bull bias + pullback to EMA + RSI not overbought
            if bull_bias and close[i] <= ema_21[i] * 1.005 and rsi_14[i] < 60:
                if trend_confirmed:
                    new_signal = SIZE
            
            # Short: 4h bear bias + pullback to EMA + RSI not oversold
            if bear_bias and close[i] >= ema_21[i] * 0.995 and rsi_14[i] > 40:
                if trend_confirmed:
                    new_signal = -SIZE
        
        else:
            # === TRANSITION REGIME: Conservative entries ===
            # Only enter with strong confluence
            if bull_bias and rsi_14[i] < 40 and close[i] <= bb_lower[i] * 1.005:
                new_signal = SIZE * 0.5
            
            if bear_bias and rsi_14[i] > 60 and close[i] >= bb_upper[i] * 0.995:
                new_signal = -SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position (strong signal)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias and chop_14[i] < 40:
                new_signal = 0.0
            if position_side < 0 and bull_bias and chop_14[i] < 40:
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