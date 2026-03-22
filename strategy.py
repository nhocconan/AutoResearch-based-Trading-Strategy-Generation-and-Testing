#!/usr/bin/env python3
"""
Experiment #294: 1d Supertrend with 1w HMA Bias and ADX Filter

Hypothesis: After analyzing 293 experiments, clear patterns emerge:
1. 4h Supertrend + 1d HMA achieved Sharpe=0.485 (#292) - BEST RESULT
2. 12h EMA + Fisher only got Sharpe=0.111 (#293) - too complex
3. RSI pullbacks consistently FAIL across all timeframes
4. Supertrend is proven to work well on crypto (less whipsaw than EMA)
5. 1d timeframe should capture major trends with fewer false signals

This strategy uses:
1. 1d Supertrend(10, 3.0) for primary trend direction (proven in #292)
2. 1w HMA(21) for major trend bias (higher than #292's 1d bias)
3. ADX(14)>15 for trend strength (loose threshold for daily to ensure trades)
4. ATR(14) for position sizing and 3.5*ATR trailing stoploss
5. Simple logic: fewer conditions = more trades = better statistical significance

Why this might beat #292 (Sharpe=0.485):
- 1d has fewer false signals than 4h (less fee drag from churn)
- 1w HMA bias is stronger filter than 1d HMA (captures major cycles)
- Supertrend alone is simpler than EMA+Fisher combination
- Daily timeframe better suited for multi-year hold periods (2021-2026)
- Looser ADX (15 vs 18) ensures >=10 trades per symbol on daily data

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.5 * ATR(14) trailing (wider for daily timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_1w_hma_adx_atr_v1"
timeframe = "1d"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    
    Formula:
    1. ATR(period)
    2. Upper Band = (high + low) / 2 + multiplier * ATR
    3. Lower Band = (high + low) / 2 - multiplier * ATR
    4. Supertrend = Lower Band if close > previous Supertrend, else Upper Band
    5. Direction = 1 if close > Supertrend (bullish), -1 if close < Supertrend (bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    direction[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        # Initial value
        if i == period:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            # If close crosses above previous supertrend, switch to lower band
            if close[i] > supertrend[i - 1]:
                supertrend[i] = max(lower_band[i], supertrend[i - 1] if not np.isnan(supertrend[i - 1]) else lower_band[i])
                direction[i] = 1
            # If close crosses below previous supertrend, switch to upper band
            elif close[i] < supertrend[i - 1]:
                supertrend[i] = min(upper_band[i], supertrend[i - 1] if not np.isnan(supertrend[i - 1]) else upper_band[i])
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i - 1]
                direction[i] = direction[i - 1]
    
    return supertrend, direction

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
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging.
    We use 15 as threshold for 1d timeframe to ensure enough trades.
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_INCREASED = 0.35  # Increased size in strong trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = major trend bias (stronger than 1d HMA)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 15 = trending market (loose for daily to ensure trades)
        trending = adx[i] > 15
        
        # === SUPERTREND DIRECTION ===
        # st_direction = 1 (bullish) or -1 (bearish)
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Strong trend = increase size
        strong_trend = adx[i] > 30
        
        # Determine position size based on volatility and trend strength
        if high_volatility:
            position_size = SIZE_REDUCED
        elif strong_trend:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        # LONG: 1w HMA bias up + Supertrend bullish + ADX confirms trend
        # Keep conditions simple to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_1w and  # 1w HMA bias bullish
            st_bullish and  # Supertrend direction bullish
            trending  # ADX confirms trending market
        )
        
        # SHORT: Mirror of long
        short_conditions = (
            bear_trend_1w and  # 1w HMA bias bearish
            st_bearish and  # Supertrend direction bearish
            trending  # ADX confirms trending market
        )
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.5 * ATR below highest close
                stoploss_price = highest_close - 3.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.5 * ATR above lowest close
                stoploss_price = lowest_close + 3.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0  # 1w trend reversed against long
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0  # 1w trend reversed against short
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit if Supertrend direction reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_bearish:
                new_signal = 0.0  # Supertrend flipped against long
            if position_side < 0 and st_bullish:
                new_signal = 0.0  # Supertrend flipped against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals