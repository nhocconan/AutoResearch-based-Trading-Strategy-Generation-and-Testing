#!/usr/bin/env python3
"""
Experiment #302: 30m Supertrend with 4h HMA Bias and RSI Pullback Entries

Hypothesis: After analyzing 298+ experiments, clear patterns emerge:
1. Supertrend + HTF HMA bias is the ONLY consistently profitable formula (#292 Sharpe=0.485)
2. Mean reversion strategies (RSI extremes, CRSI, Bollinger) ALWAYS fail on crypto
3. Complex ensembles with too many filters reduce trade count and Sharpe
4. 30m timeframe has more noise than 4h but can capture more intraday trends

This strategy adapts the winning #292 formula for 30m:
1. 30m Supertrend(ATR=10, mult=3) for primary trend direction
2. 4h HMA(21) for higher timeframe bias (proven edge from #292)
3. RSI(14) pullback entries - NOT mean reversion, but enter on pullbacks WITH trend
   - Long: Supertrend long + 4h HMA bullish + RSI(14) < 45 (pullback in uptrend)
   - Short: Supertrend short + 4h HMA bearish + RSI(14) > 55 (pullback in downtrend)
4. ADX(14) > 18 filter to avoid choppy markets (looser than 25 for more trades)
5. ATR(14) trailing stoploss at 2.5x for risk management

Why this might beat #292:
- 30m captures more intraday trends than 4h (more trade opportunities)
- RSI pullback entries catch better entry prices than pure breakout
- 4h HMA bias prevents trading against higher timeframe trend
- Simpler than failed complex ensembles (#295, #297)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_pullback_adx_atr_v1"
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
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = long (price above ST), -1 = short (price below ST)
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previous trend was long
            if close[i] > supertrend[i-1]:
                # Continue long, use lower band
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                # Flip to short
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previous trend was short
            if close[i] < supertrend[i-1]:
                # Continue short, use upper band
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            else:
                # Flip to long
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = loss_s > 0
    rs[mask] = gain_s[mask] / loss_s[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[loss_s == 0] = 100
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
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
    
    # Smooth with Wilder's method
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # === TREND STRENGTH ===
        # ADX > 18 = trending market (loose threshold for 30m to get >=10 trades)
        trending = adx[i] > 18
        strong_trend = adx[i] > 25
        
        # === RSI PULLBACK (NOT MEAN REVERSION) ===
        # Long: RSI < 45 (pullback in uptrend, not oversold)
        # Short: RSI > 55 (pullback in downtrend, not overbought)
        rsi_pullback_long = rsi[i] < 45
        rsi_pullback_short = rsi[i] > 55
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility and trend strength
        if high_volatility:
            position_size = SIZE_BASE  # Conservative in high vol
        elif strong_trend:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need Supertrend long + 4h HMA bullish + RSI pullback + ADX filter
        # These conditions work together, not conflicting
        long_conditions = (
            st_long and  # Supertrend says long
            bull_trend_4h and  # 4h HMA bias bullish
            rsi_pullback_long and  # RSI pullback (better entry price)
            trending  # ADX confirms trend
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            st_short and  # Supertrend says short
            bear_trend_4h and  # 4h HMA bias bearish
            rsi_pullback_short and  # RSI pullback (better entry price)
            trending  # ADX confirms trend
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if Supertrend reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_short:
                new_signal = 0.0  # Supertrend flipped against long
            if position_side < 0 and st_long:
                new_signal = 0.0  # Supertrend flipped against short
        
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
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