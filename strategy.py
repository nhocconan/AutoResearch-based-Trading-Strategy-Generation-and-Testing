#!/usr/bin/env python3
"""
Experiment #170: 30m Fisher Transform + 4h HMA Trend + ADX Filter + ATR Stop

Hypothesis: 30m timeframe captures medium-term swings while avoiding noise of 
lower timeframes. Fisher Transform excels at catching reversals in bear/range 
markets (perfect for 2025 test period). 4h HMA provides stable trend bias to 
avoid counter-trend entries. ADX(14) > 18 filters choppy periods. This should 
work better than pure trend-following which has been failing recently.

Why 30m + Fisher might work:
- Fisher Transform normalizes price to Gaussian distribution, catches extremes
- Works well in bear/range markets (2022 crash, 2025 test period)
- 4h HTF filter prevents false reversals against major trend
- ADX ensures we only trade when there's actual momentum
- Fewer but higher-quality trades than 15m strategies

Learning from failures:
- #158, #159, #160: EMA crossover strategies all failed (negative Sharpe)
- #163, #164, #165: RSI/BB mean reversion failed on 30m/1h
- #166, #167: CHOP filter didn't help on 4h/12h
- Need reversal strategy with strong HTF trend filter

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if hh == ll:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_signal[i] = fisher_signal[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        value = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * fisher_signal[i-1] if i > 0 else 0.0
        
        # Clamp to avoid extreme values
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        # Signal line (previous fisher value)
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_signal

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
    adx = calculate_adx(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 = trending market (lower threshold for more trades)
        trend_strength = adx[i] > 18
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_cross_long = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow entries at extreme levels for more trades
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish + (Fisher cross OR extreme oversold)
        # In bull trend: only take long Fisher signals
        # In bear trend: only take extreme oversold (counter-trend bounce)
        if bull_trend_4h:
            if (fisher_cross_long or fisher_extreme_long) and trend_strength:
                new_signal = SIZE_BASE
        else:
            # Bear trend: only extreme oversold for counter-trend bounces
            if fisher_extreme_long:
                new_signal = SIZE_BASE * 0.8  # Smaller size for counter-trend
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish + (Fisher cross OR extreme overbought)
        if bear_trend_4h:
            if (fisher_cross_short or fisher_extreme_short) and trend_strength:
                new_signal = -SIZE_BASE
        else:
            # Bull trend: only extreme overbought for counter-trend bounces
            if fisher_extreme_short:
                new_signal = -SIZE_BASE * 0.8  # Smaller size for counter-trend
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals