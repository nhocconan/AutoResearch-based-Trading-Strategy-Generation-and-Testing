#!/usr/bin/env python3
"""
Experiment #332: 30m Supertrend Pullback with 4h HMA Bias and Volume Confirmation

Hypothesis: After analyzing failures #320-331, complex regime filters and EMA crossovers
are failing on 30m. The successful pattern from #329 (MACD momentum, Sharpe=0.311) suggests
momentum-based entries work better than mean reversion on this timeframe.

This strategy combines:
1. 4h HMA(21) for directional bias (proven edge from multiple experiments)
2. 30m Supertrend(10,3) for trend direction and stoploss
3. RSI(14) pullback filter - enter on pullback NOT breakout (RSI 40-60 zone)
4. Volume confirmation - volume > 0.8 * 20-bar avg (avoid low liquidity entries)
5. ATR(14) trailing stoploss at 2.5x for additional protection

Key differences from failed strategies:
- Supertrend instead of EMA crossover (better trend following)
- RSI pullback (40-60) instead of extremes (avoid over-filtering)
- Volume filter is soft confirmation, not hard requirement
- Loose ADX > 15 (not > 25) to ensure trade generation
- Position size 0.25 base, 0.30 strong trend

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: Supertrend flip + 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_pullback_volume_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize arrays
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish (price above supertrend), -1 = bearish
    
    # First valid bar
    supertrend[period] = upper_band[period]
    
    for i in range(period + 1, n):
        if direction[i-1] == 1:
            # Previously bullish
            if close[i] > lower_band[i]:
                # Stay bullish
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                # Flip to bearish
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously bearish
            if close[i] < upper_band[i]:
                # Stay bearish
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            else:
                # Flip to bullish
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume moving average
    volume_s = pd.Series(volume)
    volume_avg = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = primary directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        # st_direction = 1 means bullish (price above supertrend)
        # st_direction = -1 means bearish (price below supertrend)
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Check for supertrend flip (entry signal)
        st_flip_bull = st_bullish and (i > 0 and st_direction[i-1] == -1)
        st_flip_bear = st_bearish and (i > 0 and st_direction[i-1] == 1)
        
        # === RSI PULLBACK FILTER ===
        # For longs: RSI between 40-60 (pullback, not overbought)
        # For shorts: RSI between 40-60 (pullback, not oversold)
        rsi_pullback_long = 40 <= rsi[i] <= 65
        rsi_pullback_short = 35 <= rsi[i] <= 60
        
        # === ADX TREND STRENGTH ===
        # ADX > 15 = minimal trending (loose for trade generation)
        trending = adx[i] > 15
        strong_trend = adx[i] > 25
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.8 * 20-bar average (soft filter)
        vol_confirmed = volume[i] > 0.8 * volume_avg[i] if not np.isnan(volume_avg[i]) else True
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # Boost size for strong trend + 4h alignment
        if strong_trend and bull_trend_4h:
            position_size = SIZE_STRONG
        elif strong_trend and bear_trend_4h:
            position_size = SIZE_STRONG
        
        # LONG: 4h bias up + Supertrend bullish + RSI pullback + trending
        # Allow entry on supertrend flip OR if already in bullish state with pullback
        long_conditions = (
            bull_trend_4h and
            st_bullish and
            rsi_pullback_long and
            trending
        )
        
        # SHORT: 4h bias down + Supertrend bearish + RSI pullback + trending
        short_conditions = (
            bear_trend_4h and
            st_bearish and
            rsi_pullback_short and
            trending
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Primary: Supertrend flip against position
        # Secondary: ATR trailing stop at 2.5x
        
        if in_position and position_side != 0:
            stoploss_triggered = False
            
            if position_side > 0:
                # Update highest close for trailing stop
                if close[i] > highest_close:
                    highest_close = close[i]
                
                # Supertrend flip to bearish
                if st_bearish:
                    stoploss_triggered = True
                
                # ATR trailing stop
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
                
                # 4h trend reversal
                if bear_trend_4h:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for trailing stop
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                
                # Supertrend flip to bullish
                if st_bullish:
                    stoploss_triggered = True
                
                # ATR trailing stop
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
                
                # 4h trend reversal
                if bull_trend_4h:
                    stoploss_triggered = True
            
            if stoploss_triggered:
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
                # Position reversal
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