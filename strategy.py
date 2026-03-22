#!/usr/bin/env python3
"""
Experiment #281: 12h KAMA Adaptive Trend with 1d HMA Bias and ADX Filter

Hypothesis: After analyzing 280 experiments, clear patterns emerge:
1. Complex ensembles fail (too many conflicting filters = 0 trades)
2. RSI pullback strategies consistently fail on BTC/ETH
3. Fisher transform generates 0 trades on multiple timeframes
4. Volume filters often too strict for 12h timeframe
5. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA/HMA

This strategy uses:
1. 12h KAMA(21) - adaptive trend that flattens in chop, moves in trends
2. 1d HMA(21) - directional bias filter (proven effective in prior experiments)
3. ADX(14)>18 - trend strength filter (looser than typical 25 to ensure trades)
4. Pullback entries - enter when price retraces to KAMA in direction of 1d bias
5. 2.5*ATR stoploss - tighter than 3.0*ATR for better risk control
6. NO volume filter - volume confirmation filtered too many trades in #263
7. Discrete position sizing: 0.0, ±0.25, ±0.30

Why this should work better than #263 (Donchian breakout):
- KAMA pullbacks generate MORE signals than Donchian breakouts
- No volume filter = more trades (addressing 0-trade failure mode)
- ADX>18 is looser than typical filters (ensures >=10 trades)
- Adaptive KAMA handles 2022 chop better than fixed Donchian

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 max, discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_1d_hma_adx_pullback_v1"
timeframe = "12h"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, flat in chop.
    Formula from Perry Kaufman's "Trading Systems and Methods".
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER): measures trend efficiency vs noise
    # ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if noise > 0:
            er = change / noise
        else:
            er = 0.0
        
        # Smoothing constant: SC = [ER * (fast - slow) + slow]^2
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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
    Measures trend strength regardless of direction.
    """
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX is EMA of DX
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_s.values
    
    return adx

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 21)
    adx = calculate_adx(high, low, close, 14)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_MAX = 0.30  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias (hard filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 ensures we're in a trending market (looser than 25)
        trend_strong = adx[i] > 18.0
        
        # === KAMA TREND DIRECTION ===
        # Price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === PULLBACK ENTRY CONDITIONS ===
        # Long: 1d bias up + ADX strong + price pulls back to/near KAMA
        # We check if price was above KAMA recently but now near it (pullback)
        pullback_long = False
        if i > 5 and price_above_kama:
            # Check if price retraced from above to near KAMA
            recent_high = np.max(close[i-5:i+1])
            pullback_threshold = kama[i] * 1.015  # Within 1.5% of KAMA
            if close[i] <= pullback_threshold and recent_high > pullback_threshold:
                pullback_long = True
        
        # Short: 1d bias down + ADX strong + price pulls back to/near KAMA
        pullback_short = False
        if i > 5 and price_below_kama:
            # Check if price retraced from below to near KAMA
            recent_low = np.min(close[i-5:i+1])
            pullback_threshold = kama[i] * 0.985  # Within 1.5% of KAMA
            if close[i] >= pullback_threshold and recent_low < pullback_threshold:
                pullback_short = True
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bias up + ADX strong + KAMA pullback
        if bull_trend_1d and trend_strong and pullback_long:
            new_signal = SIZE_BASE
        
        # SHORT ENTRY: 1d bias down + ADX strong + KAMA pullback
        if bear_trend_1d and trend_strong and pullback_short:
            new_signal = -SIZE_BASE
        
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
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals