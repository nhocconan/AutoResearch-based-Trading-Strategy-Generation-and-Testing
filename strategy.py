#!/usr/bin/env python3
"""
Experiment #356: 30m Volatility Breakout with 4h KAMA Trend + ADX + Volume

Hypothesis: After analyzing 355 failed experiments, the pattern is clear:
1. Simple trend following (EMA/HMA crossover) fails on BTC/ETH in bear/range markets
2. Pure mean reversion misses major moves and gets stopped out in trends
3. What works: VOLATILITY BREAKOUTS with HTF trend bias + volume confirmation

This strategy combines:
1. 4h KAMA (Kaufman Adaptive MA) - adapts to market efficiency, less whipsaw than EMA/HMA
   - KAMA uses Efficiency Ratio to adjust smoothing constant
   - Fast in trends, slow in chop - perfect for regime adaptation

2. 30m Donchian(20) Breakout - captures volatility expansion
   - Break above 20-period high = momentum starting
   - Break below 20-period low = downside momentum

3. 30m ADX(14) Filter - confirm trend strength
   - ADX > 20 = trending market (allow breakouts)
   - ADX < 20 = ranging (skip to avoid whipsaw)

4. 30m Volume Confirmation - taker_buy_volume ratio
   - Volume ratio > 1.2 = conviction behind breakout
   - Filters out low-volume fake breakouts

5. ATR(14)*2.5 Trailing Stop - protect capital
   - Signal → 0 when price moves 2.5*ATR against position

6. Position Sizing: 0.25 discrete (conservative for 30m volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 30m should work:
- Faster than 4h/12h (more trade opportunities)
- Slower than 5m/15m (less noise, fewer false signals)
- 4h KAMA provides stable adaptive trend bias
- Volume filter reduces fake breakouts significantly
- Should generate 40-80 trades/year per symbol (enough for stats)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_breakout_4h_kama_adx_vol_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - fast in trends, slow in chop.
    
    Efficiency Ratio (ER) = |Price Change| / Sum of |Individual Price Changes|
    SC (Smoothing Constant) = (ER * (fast_sc - slow_sc) + slow_sc)^2
    fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + er_period:
        return kama
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.nan
    
    sum_price_change = np.zeros(n)
    for i in range(er_period, n):
        sum_price_change[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    for i in range(er_period, n):
        if sum_price_change[i] > 1e-10:
            er[i] = price_change[i] / sum_price_change[i]
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate SC
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_donchian_channels(high, low, period=20):
    """
    Calculate Donchian Channels.
    Upper = highest high of last N periods
    Lower = lowest low of last N periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    kama_4h = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.zeros(n)
    for i in range(1, n):
        if volume[i] > 1e-10:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
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
        
        if np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4h KAMA TREND BIAS ===
        # KAMA adapts to market efficiency - more reliable than static MA
        bull_trend_4h = close[i] > kama_4h_aligned[i]
        bear_trend_4h = close[i] < kama_4h_aligned[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 20 = trending market (allow breakout entries)
        trending_market = adx[i] > 20
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 0.55 = buying pressure, < 0.45 = selling pressure
        volume_buying = volume_ratio[i] > 0.55
        volume_selling = volume_ratio[i] < 0.45
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long breakout: price breaks above previous Donchian upper
        long_breakout = close[i] > donchian_upper[i-1] if i > 0 else False
        
        # Short breakout: price breaks below previous Donchian lower
        short_breakout = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Breakout + 4h bullish bias + ADX confirms + volume supports
        if long_breakout and bull_trend_4h and trending_market and volume_buying:
            new_signal = SIZE
        
        # SHORT ENTRY: Breakout + 4h bearish bias + ADX confirms + volume supports
        elif short_breakout and bear_trend_4h and trending_market and volume_selling:
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
        if in_position and adx[i] < 18:
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