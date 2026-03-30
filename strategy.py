#!/usr/bin/env python3
"""
Experiment #022: Donchian Breakout + Williams Alligator + TRIX Momentum (12h)

HYPOTHESIS: Combine proven price channel breakout with trend detection systems:
1. Donchian(20) breakout - price structure (proven winner in DB)
2. Williams Alligator - trend direction confirmation
3. TRIX momentum - entry timing filter
4. Volume spike - breakout validation

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above 20-bar high + Alligator lines aligned upward + TRIX > 0
- Bear: Price breaks below 20-bar low + Alligator lines aligned downward + TRIX < 0
- Range: No Donchian breakout = no trade (avoids whipsaws)
- 12h timeframe reduces noise vs 4h, fewer trades = less fee drag

KEY INSIGHT from DB: Best performers (Sharpe 1.3-1.8) use price channel breakout
(Donchian/Camarilla) + volume confirmation + regime filter. Keep it simple.
The previous #021 had too many conditions (Ichimoku + Alligator + volume + ADX + HTF)
= 0 trades. This version uses 3 core conditions.

TARGET: 75-250 total trades over 4 years (15-50/year on 12h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_alligator_trix_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_trix(close, period=14, signal=9):
    """
    TRIX - Triple EMA Oscillator
    - TRIX > 0 + rising = bullish momentum
    - TRIX < 0 + falling = bearish momentum
    - Signal line crossover = entry timing
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(period * 3, n):
        if ema3[i-1] != 0:
            trix[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 100
    
    # Signal line = EMA of TRIX
    trix_series = pd.Series(trix)
    trix_ema = trix_series.ewm(span=signal, min_periods=signal, adjust=False).mean().values
    
    return trix, trix_ema

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Williams Alligator: SMMA-based trend detection
    - Jaw (blue): 13-period SMMA of high
    - Teeth (red): 8-period SMMA of close
    - Lips (green): 5-period SMMA of low
    """
    n = len(close)
    
    # SMMA function
    def smma(data, period):
        result = np.zeros(n)
        if period <= 0:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, n):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    jaw = smma(high, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(low, lips_period)
    
    return jaw, teeth, lips

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20 period as per DB winners"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_adx(high, low, close, period=14):
    """ADX for trend strength filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Donchian for HTF trend direction ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # 1d Donchian(20) for HTF direction
    df_1d_upper = np.full(len(df_1d), np.nan)
    df_1d_lower = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        df_1d_upper[i] = np.max(df_1d_high[i - 19:i + 1])
        df_1d_lower[i] = np.min(df_1d_low[i - 19:i + 1])
    
    # HTF: Price above/below 1d Donchian = trend direction
    htf_bullish = df_1d_close > df_1d_upper
    htf_bearish = df_1d_close < df_1d_lower
    
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20) - proven price channel from DB
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # TRIX momentum
    trix, trix_signal = calculate_trix(close, period=14, signal=9)
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Precompute bar-by-bar conditions ===
    # Donchian breakout signals
    donchian_breakout_up = np.zeros(n, dtype=bool)
    donchian_breakout_down = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if not np.isnan(donchian_upper[i]) and not np.isnan(donchian_upper[i-1]):
            # Price closes above yesterday's upper band
            donchian_breakout_up[i] = close[i] > donchian_upper[i-1]
            donchian_breakout_down[i] = close[i] < donchian_lower[i-1]
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # TRIX needs ~42, Alligator needs 13, Donchian needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        # === ALLIGATOR TREND DETECTION ===
        # Trending up: jaw > teeth > lips (price action above all lines)
        bull_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and (close[i] > jaw[i])
        # Trending down: jaw < teeth < lips (price action below all lines)
        bear_alligator = (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and (close[i] < jaw[i])
        
        # Alligator spread (trend strength indicator)
        jaw_teeth_spread = jaw[i] - teeth[i] if bull_alligator else teeth[i] - jaw[i]
        alligator_strong = abs(jaw_teeth_spread) > atr_14[i] * 0.15
        
        # === TRIX MOMENTUM ===
        trix_bullish = trix[i] > trix_signal[i]
        trix_bearish = trix[i] < trix_signal[i]
        trix_momentum = trix[i] - trix_signal[i]  # positive = bullish momentum
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.6
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 20
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Donchian breakout + Bullish Alligator + TRIX bullish + volume spike
            # Relaxed conditions to ensure trades (previous version had 0 trades)
            long_donchian = donchian_breakout_up[i] and close[i] > donchian_mid[i]
            long_alligator = bull_alligator and alligator_strong
            long_trix = trix_bullish and trix_momentum > 0.02  # minimum momentum threshold
            long_vol = vol_ratio[i] > 1.4  # relaxed volume
            
            # Entry: Donchian breakout + Alligator confirmation (TRIX optional if HTF strong)
            if long_donchian and long_alligator and (long_trix or htf_bull) and (long_vol or htf_bull):
                desired_signal = SIZE
            
            # SHORT: Donchian breakdown + Bearish Alligator + TRIX bearish + volume spike
            short_donchian = donchian_breakout_down[i] and close[i] < donchian_mid[i]
            short_alligator = bear_alligator and alligator_strong
            short_trix = trix_bearish and trix_momentum < -0.02
            short_vol = vol_ratio[i] > 1.4
            
            if short_donchian and short_alligator and (short_trix or htf_bear) and (short_vol or htf_bear):
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if Alligator turns bearish
                if bear_alligator and not bull_alligator:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if Alligator turns bullish
                if bull_alligator and not bear_alligator:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals