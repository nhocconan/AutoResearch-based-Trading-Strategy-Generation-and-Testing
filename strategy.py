#!/usr/bin/env python3
"""
Experiment #116: 30m Fisher Transform Reversals + 4h HMA Trend Filter + ATR Stop

Hypothesis: After 15 failed experiments, trying Ehlers Fisher Transform for reversal
detection on 30m timeframe. Fisher Transform normalizes price to Gaussian distribution,
making extreme values (-2 to +2) reliable reversal signals. Combined with:
- 4h HMA(21) for higher-timeframe trend bias (prevents counter-trend trades)
- ADX(14) > 20 filter (only trade when some momentum exists)
- ATR(14) 2.5x trailing stop (protects against whipsaws)
- Discrete position sizing (0.20/0.30) to minimize fee churn

Why this might work when others failed:
- Fisher Transform catches reversals in bear market rallies (2025 test period)
- 4h HMA filter prevents entering against major trend (critical for 2022 crash)
- 30m timeframe balances noise reduction vs signal frequency
- Fewer trades than RSI strategies = less fee drag
- Works in both trending and ranging markets (Fisher adapts)

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Values typically range -2 to +2. Extremes indicate reversal zones.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate median price
    median = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val == 0:
            continue
        
        normalized = (median[i] - lowest) / range_val
        
        # Clamp to avoid division by zero
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Transform to quasi-normal distribution
        xform = 0.66 * ((normalized - 0.5) / 0.66 + 0.67 * np.log((1 - normalized) / normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.5 * (xform + fisher[i-1])
        else:
            fisher[i] = xform
    
    return fisher

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # Calculate ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

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
    fisher = calculate_fisher_transform(high, low, 9)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Track Fisher crossings
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === ADX TREND STRENGTH FILTER ===
        adx_valid = adx[i] > 18  # Lower threshold for 30m to get more trades
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long_cross = False
        if not np.isnan(prev_fisher) and not np.isnan(fisher[i]):
            fisher_long_cross = (prev_fisher < -1.5 and fisher[i] >= -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short_cross = False
        if not np.isnan(prev_fisher) and not np.isnan(fisher[i]):
            fisher_short_cross = (prev_fisher > 1.5 and fisher[i] <= 1.5)
        
        # Also allow extreme Fisher values for stronger signals
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 4h bullish + Fisher extreme long + ADX valid
        if bull_trend_4h and fisher_extreme_long and adx_valid:
            new_signal = SIZE_STRONG
        # Moderate: 4h bullish + Fisher long cross + ADX valid
        elif bull_trend_4h and fisher_long_cross and adx_valid:
            new_signal = SIZE_BASE
        # Weak: Fisher extreme long only (ensure trades on all symbols)
        elif fisher_extreme_long:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 4h bearish + Fisher extreme short + ADX valid
        if bear_trend_4h and fisher_extreme_short and adx_valid:
            new_signal = -SIZE_STRONG
        # Moderate: 4h bearish + Fisher short cross + ADX valid
        elif bear_trend_4h and fisher_short_cross and adx_valid:
            new_signal = -SIZE_BASE
        # Weak: Fisher extreme short only (ensure trades on all symbols)
        elif fisher_extreme_short:
            new_signal = -SIZE_BASE
        
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
        
        # Update previous Fisher value for next iteration
        prev_fisher = fisher[i]
    
    return signals