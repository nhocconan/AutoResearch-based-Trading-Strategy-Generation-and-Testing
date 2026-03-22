#!/usr/bin/env python3
"""
Experiment #312: 1d HMA Trend + RSI Pullback with 1w Meta-Bias and ATR Stoploss

Hypothesis: Daily timeframe needs simpler logic with LOOSE entry conditions to generate
sufficient trades (>=10 on train, >=3 on test). Analysis of #300 (Sharpe=0.358) and
#304 (Sharpe=0.367) shows HTF HMA bias + trend following works. This strategy:

1. 1d HMA(21) = primary trend direction (price above = bull, below = bear)
2. 1w HMA(21) = meta-trend confirmation (soft filter, boosts confidence)
3. RSI(7) pullback entries = long when RSI<45 in uptrend, short when RSI>55 in downtrend
4. ADX(14)>15 = minimal momentum filter (very loose for trade generation)
5. ATR(14) trailing stoploss at 2.5x (proven from successful strategies)
6. Discrete position sizing: 0.25 base, 0.30 with 1w confirmation

Key improvements from #311:
- RSI pullback instead of EMA crossover (catches dip entries in trend)
- Lower ADX threshold (15 vs 20) for more trades on 1d
- Simpler position tracking logic
- 1d primary TF as required by experiment

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_bias_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
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
    dx = dx.fillna(0.0)
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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    rsi = calculate_rsi(close, 7)  # Faster RSI for more signals
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(50, n):  # Start at 50 to ensure indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = meta-trend direction
        bull_meta = close[i] > hma_1w_aligned[i]
        bear_meta = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND ===
        # 1d HMA = primary directional bias
        bull_trend = close[i] > hma_1d[i]
        bear_trend = close[i] < hma_1d[i]
        
        # === MOMENTUM ===
        # ADX > 15 = minimal trending (loose for trade generation)
        trending = adx[i] > 15
        
        # === RSI PULLBACK ENTRY ===
        # Long: in uptrend + RSI pulled back (oversold in trend)
        rsi_long_pullback = rsi[i] < 45
        # Short: in downtrend + RSI rallied (overbought in trend)
        rsi_short_pullback = rsi[i] > 55
        
        # === EXTREME RSI REVERSAL ===
        # Long: very oversold regardless of trend (mean reversion)
        rsi_extreme_long = rsi[i] < 25
        # Short: very overbought regardless of trend (mean reversion)
        rsi_extreme_short = rsi[i] > 75
        
        # Determine position size
        if bull_meta and bull_trend:
            position_size = SIZE_STRONG
        elif bear_meta and bear_trend:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: uptrend + RSI pullback OR extreme oversold
        long_conditions = (
            (bull_trend and rsi_long_pullback and trending) or
            rsi_extreme_long
        )
        
        # SHORT: downtrend + RSI pullback OR extreme overbought
        short_conditions = (
            (bear_trend and rsi_short_pullback and trending) or
            rsi_extreme_short
        )
        
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === RSI REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 70:
                new_signal = 0.0  # Take profit on overbought
            if position_side < 0 and rsi[i] < 30:
                new_signal = 0.0  # Take profit on oversold
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals