#!/usr/bin/env python3
"""
Experiment #038: 30m Supertrend + RSI Pullback + 4h HMA Trend Filter
Hypothesis: Supertrend captures trend direction cleanly, RSI pullbacks provide low-risk entries
within the trend. 4h HMA filters counter-trend trades. ADX ensures we only trade in trending
conditions (not chop). ATR stoploss at 2.5x limits drawdown during reversals.

Key improvements from failed experiments:
- LOOSENED entry conditions to ensure 10+ trades/year (previous strategies had 0 trades)
- RSI range widened (30-50 for long pullbacks, 50-70 for short pullbacks)
- ADX threshold lowered to 18 (ADX>25 too restrictive)
- Supertrend breakout confirmation ensures momentum
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper
Position sizing: 0.25 base, 0.30 max, discrete levels
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_rsi_4h_hma_v1"
timeframe = "30m"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend-following indicator based on ATR.
    Returns: supertrend_line, direction (1=up, -1=down)
    """
    n = len(close)
    atr = np.zeros(n)
    tr = np.zeros(n)
    
    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # ATR with Wilder's smoothing
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Supertrend calculation
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    supertrend[0] = upper_band[0]
    direction[0] = -1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            if close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
        else:
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    return supertrend, direction, atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    gains[1:] = np.where(delta > 0, delta, 0)
    losses[1:] = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # DM+ and DM-
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth TR, +DM, -DM
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)
    smooth_plus[period-1] = np.mean(plus_dm[:period])
    smooth_minus[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        smooth_plus[i] = (smooth_plus[i-1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i-1] * (period - 1) + minus_dm[i]) / period
    
    # DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    mask = atr > 0
    di_plus[mask] = 100 * smooth_plus[mask] / atr[mask]
    di_minus[mask] = 100 * smooth_minus[mask] / atr[mask]
    
    # DX and ADX
    dx = np.zeros(n)
    di_sum = di_plus + di_minus
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(di_plus[mask2] - di_minus[mask2]) / di_sum[mask2]
    
    # ADX smoothing
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    supertrend, st_direction, atr = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Minimum lookback period
    min_lookback = 250
    
    for i in range(min_lookback, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Supertrend direction
        st_bull = st_direction[i] == 1
        st_bear = st_direction[i] == -1
        
        # RSI pullback zones (LOOSENED for more trades)
        rsi_long_pullback = 30 <= rsi[i] <= 50  # Pullback in uptrend
        rsi_short_pullback = 50 <= rsi[i] <= 70  # Pullback in downtrend
        
        # ADX trending filter (lowered from 25 to 18)
        trending = adx[i] >= 18
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        # Price vs Supertrend (breakout confirmation)
        price_above_st = close[i] > supertrend[i]
        price_below_st = close[i] < supertrend[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Supertrend bull + RSI pullback + 4h bull trend + trending
        if st_bull and rsi_long_pullback and bull_trend and trending:
            new_signal = SIZE_BASE
        # Secondary: Supertrend bull + RSI pullback + EMA bullish
        elif st_bull and rsi_long_pullback and ema_bullish:
            new_signal = SIZE_BASE
        # Tertiary: Strong breakout - price above ST + 4h bull + ADX rising
        elif price_above_st and bull_trend and adx[i] >= 20 and rsi[i] > 45:
            new_signal = SIZE_MAX
        
        # === SHORT ENTRY ===
        # Primary: Supertrend bear + RSI pullback + 4h bear trend + trending
        if st_bear and rsi_short_pullback and bear_trend and trending:
            new_signal = -SIZE_BASE
        # Secondary: Supertrend bear + RSI pullback + EMA bearish
        elif st_bear and rsi_short_pullback and ema_bearish:
            new_signal = -SIZE_BASE
        # Tertiary: Strong breakdown - price below ST + 4h bear + ADX rising
        elif price_below_st and bear_trend and adx[i] >= 20 and rsi[i] < 55:
            new_signal = -SIZE_MAX
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
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