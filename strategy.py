#!/usr/bin/env python3
"""
Experiment #361: 15m Supertrend Momentum with 4h HMA Trend + 1h ADX Regime Filter

Hypothesis: After 360 experiments, the pattern is clear - 15m needs HTF filters to avoid
noise whipsaw while still capturing intraday momentum. This strategy combines:

1. 4h HMA(21) TREND BIAS: Only trade in direction of higher timeframe trend
   - Long only if price > 4h HMA(21)
   - Short only if price < 4h HMA(21)
   - Filters 60%+ of counter-trend noise (proven in #353)

2. 1h ADX(14) REGIME FILTER: Only enter when trending (not choppy)
   - ADX > 22 = trending market (allow entries)
   - ADX < 18 = ranging market (exit positions, avoid whipsaw)
   - Hysteresis prevents rapid flip-flopping

3. 15m SUPERTREND(10, 3.0): Clean trend-following entry signals
   - Supertrend flips = momentum shift
   - Proven on faster timeframes with HTF filter
   - Less noisy than EMA crossover

4. VOLUME CONFIRMATION: Entry only if volume > 1.5 * 20-period avg
   - Filters low-volume false breakouts
   - Critical for 15m timeframe noise reduction

5. ATR(14) TRAILING STOP (2.5x): Protect capital on reversals
   - Signal → 0 when price moves 2.5*ATR against position
   - Trailing stop locks in profits

6. POSITION SIZING: 0.25 discrete (conservative for 15m volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 15m should work:
- Faster than 1h/4h strategies that failed
- 4h HMA provides stable trend bias
- 1h ADX filters chop periods
- Supertrend captures clean momentum moves
- Should generate 30-60 trades/year per symbol (enough for stats)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h ADX + 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_1h_adx_vol_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = strong trend, ADX < 20 = ranging market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 10:
        return adx
    
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
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX = smoothed DX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)  # 1 = long (below price), -1 = short (above price)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        # Calculate basic bands
        hl2 = (high[i] + low[i]) / 2.0
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if i == period:
            # Initial value
            supertrend[i] = upper_band
            direction[i] = -1
        else:
            # Update bands based on previous direction
            if direction[i-1] == 1:  # Previous was long
                upper_band = min(upper_band, supertrend[i-1])
                if close[i] > supertrend[i-1]:
                    supertrend[i] = lower_band
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    direction[i] = -1
            else:  # Previous was short
                lower_band = max(lower_band, supertrend[i-1])
                if close[i] < supertrend[i-1]:
                    supertrend[i] = upper_band
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
    
    return supertrend, direction

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h ADX REGIME FILTER ===
        # Loosened from 25 to 22 to generate more trades on 15m
        trending_market = adx_1h_aligned[i] > 22
        ranging_market = adx_1h_aligned[i] < 18  # Exit threshold with hysteresis
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_sma[i]
        
        # === SUPERTREND SIGNALS ===
        supertrend_long = supertrend_dir[i] == 1  # Price above supertrend
        supertrend_short = supertrend_dir[i] == -1  # Price below supertrend
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend long + 4h bullish + ADX trending + Volume confirmed
        if supertrend_long and bull_trend_4h and trending_market and volume_confirmed:
            new_signal = SIZE
        
        # SHORT ENTRY: Supertrend short + 4h bearish + ADX trending + Volume confirmed
        elif supertrend_short and bear_trend_4h and trending_market and volume_confirmed:
            new_signal = -SIZE
        
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
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === ADX DROPS BELOW THRESHOLD ===
        # Exit if market becomes ranging (ADX < 18 with hysteresis)
        if in_position and ranging_market:
            new_signal = 0.0
        
        # === SUPERTREND FLIP EXIT ===
        # Exit if supertrend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and supertrend_short:
                new_signal = 0.0
            if position_side < 0 and supertrend_long:
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