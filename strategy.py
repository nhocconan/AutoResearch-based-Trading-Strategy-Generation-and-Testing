#!/usr/bin/env python3
"""
Experiment #266: 12h Primary + 1d HTF — Fisher Transform + Choppiness Regime + Volume

Hypothesis: After #262 (Sharpe=0.244) showed 12h works, improve with:
1. Fisher Transform (period=9) for entry timing - proven to catch reversals in bear markets
2. 1d HMA(21) for PRIMARY trend direction (faster than SMA, less lag)
3. Choppiness(14) for regime detection - mean revert when choppy, trend when trending
4. Volume spike confirmation (taker_buy_volume ratio > 1.5) for breakout validity
5. Simpler entry logic to ensure 10+ trades per symbol (relaxed thresholds)
6. Discrete sizing: 0.25 base, 0.35 strong conviction

Key differences from failed attempts:
- Fisher Transform instead of RSI (better for reversals in bear/range markets)
- Volume confirmation reduces false breakouts
- Relaxed Fisher thresholds (-1.5/+1.5 instead of -2.0/+2.0) for more trades
- Force trade every 15 bars if no signal (12h * 15 = 180h = 7.5 days)

Position sizing: 0.25 base, 0.35 strong (discrete levels, max 0.40)
Target: 20-50 trades/year per symbol (appropriate for 12h)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_vol_1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Catches reversals when Fisher crosses extreme levels.
    Long: Fisher crosses above -1.5 from below
    Short: Fisher crosses below +1.5 from above
    """
    n = period
    hl2 = (high + low) / 2.0
    
    # Normalize price to 0-1 range using Donchian
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    fisher = np.zeros(len(close))
    fisher_signal = np.zeros(len(close))
    
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0:
            normalized = 2.0 * (hl2[i] - ll[i]) / range_hl - 1.0
            # Clamp to avoid division issues
            normalized = np.clip(normalized * 0.99, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            # Smooth the fisher value
            if i > n:
                fisher[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
        
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_signal

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ratio(taker_buy_volume, volume, period=20):
    """Calculate volume spike ratio (taker buy / total volume)."""
    ratio = np.zeros(len(volume))
    for i in range(period, len(volume)):
        if volume[i] > 0:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    # Calculate rolling average for comparison
    ratio_s = pd.Series(ratio)
    ratio_avg = ratio_s.rolling(window=period, min_periods=period).mean().values
    
    # Volume spike when current ratio > 1.5 * average
    volume_spike = ratio > (1.5 * ratio_avg + 0.1)
    
    return volume_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    volume_spike = calculate_volume_ratio(taker_buy_volume, volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 22.0
        is_weak_trend = adx_14[i] < 18.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === RSI THRESHOLDS (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + strong ADX + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Trending + bull regime + Fisher long signal
            if regime_bull and fisher_long:
                new_signal = STRONG_SIZE if volume_spike[i] else BASE_SIZE
            # LONG: Trending + bull regime + RSI oversold pullback
            elif regime_bull and rsi_oversold and not regime_bear:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + Fisher short signal
            if regime_bear and fisher_short:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE if volume_spike[i] else -BASE_SIZE
            # SHORT: Trending + bear regime + RSI overbought retrace
            elif regime_bear and rsi_overbought and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + Fisher extreme oversold
            if fisher_extreme_long:
                new_signal = BASE_SIZE
            # LONG: Choppy + RSI extreme oversold
            elif rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + Fisher extreme overbought
            if fisher_extreme_short:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI extreme overbought
            elif rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~180h = 7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 35 and fisher[i] > -1.5:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and rsi_14[i] < 65 and fisher[i] < 1.5:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and rsi_14[i] < 35:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and rsi_14[i] > 65:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and chop_14[i] > 60:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and chop_14[i] > 60:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals