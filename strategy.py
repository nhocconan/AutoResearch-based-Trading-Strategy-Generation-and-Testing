#!/usr/bin/env python3
"""
Experiment #016: 4h Choppiness Regime + 1d HMA Trend + Fisher Transform Reversals
Hypothesis: BTC/ETH spend 60-70% of time in ranging markets. Use Choppiness Index (CHOP)
to detect regime, then apply DIFFERENT logic:
- RANGE (CHOP > 61.8): Mean reversion via Fisher Transform + RSI extremes
- TREND (CHOP < 38.2): Breakout following via Donchian + ADX confirmation
- 1d HMA via mtf_data provides overall bias (only trade with HTF trend in trend regime)

Key innovations:
1. Choppiness Index regime filter - adapts strategy to market state (range vs trend)
2. Ehlers Fisher Transform - catches reversals with less lag than RSI
3. 4h timeframe - captures multi-day swings without 12h/1d lag
4. Asymmetric sizing: 0.30 in trend regime, 0.20 in range regime (less risk in chop)
5. 2.5*ATR trailing stoploss on all positions

Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_1d_hma_fisher_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    adx = np.zeros(n)
    adx[:] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    mask = tr_smooth > 0
    di_plus[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (di_plus + di_minus) > 0
    dx[mask2] = 100 * np.abs(di_plus[mask2] - di_minus[mask2]) / (di_plus[mask2] + di_minus[mask2])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2:] = adx_raw[period * 2:]
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals when Fisher crosses extreme levels (-1.5, +1.5).
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Avoid division by zero
        if highest == lowest:
            fisher[i] = fisher[i - 1] if i > period else 0.0
            continue
        
        # Normalize price to 0-1 range
        value = (close[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid extremes
        value = np.clip(value, 0.001, 0.999)
        
        # Fisher transformation
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i > period:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures if market is trending or ranging.
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30  # Higher conviction in trend regime
    SIZE_RANGE = 0.20  # Lower conviction in choppy regime
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - determines overall market direction
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # Choppiness Index regime detection
        ranging_regime = chop[i] > 61.8  # High choppiness = range
        trending_regime = chop[i] < 38.2  # Low choppiness = trend
        neutral_regime = not ranging_regime and not trending_regime
        
        # Fisher Transform signals (mean reversion)
        fisher_long = fisher[i] < -1.5 and fisher_signal[i] >= -1.5  # Cross above -1.5
        fisher_short = fisher[i] > 1.5 and fisher_signal[i] <= 1.5  # Cross below +1.5
        
        # Fisher extreme levels (reversal zones)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_extreme_oversold = rsi[i] < 20
        rsi_extreme_overbought = rsi[i] > 80
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Bollinger mean reversion
        price_below_lower = close[i] < bb_lower[i] * 1.005
        price_above_upper = close[i] > bb_upper[i] * 0.995
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        new_signal = 0.0
        
        # === RANGE REGIME (CHOP > 61.8): Mean Reversion ===
        if ranging_regime:
            # Long: Fisher oversold + RSI oversold + near lower BB
            if fisher_oversold and rsi_oversold and price_below_lower:
                new_signal = SIZE_RANGE
            # Long: Fisher cross above -1.5 in bull HTF regime
            elif fisher_long and bull_regime:
                new_signal = SIZE_RANGE
            # Short: Fisher overbought + RSI overbought + near upper BB
            elif fisher_overbought and rsi_overbought and price_above_upper:
                new_signal = -SIZE_RANGE
            # Short: Fisher cross below +1.5 in bear HTF regime
            elif fisher_short and bear_regime:
                new_signal = -SIZE_RANGE
        
        # === TREND REGIME (CHOP < 38.2): Trend Following ===
        elif trending_regime:
            # Long: Donchian breakout + ADX strong + bull HTF regime
            if breakout_long and strong_trend and bull_regime:
                new_signal = SIZE_TREND
            # Long: Pullback to EMA50 in bull regime + Fisher turning up
            elif bull_regime and close[i] > ema_50[i] and fisher[i] > fisher_signal[i] if not np.isnan(fisher_signal[i]) else False:
                if close[i] > ema_50[i] and (not np.isnan(fisher_signal[i]) and fisher[i] > fisher_signal[i]):
                    new_signal = SIZE_TREND
            # Short: Donchian breakout + ADX strong + bear HTF regime
            elif breakout_short and strong_trend and bear_regime:
                new_signal = -SIZE_TREND
            # Short: Pullback to EMA50 in bear regime + Fisher turning down
            elif bear_regime and close[i] < ema_50[i]:
                if not np.isnan(fisher_signal[i]) and fisher[i] < fisher_signal[i]:
                    new_signal = -SIZE_TREND
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Conservative ===
        elif neutral_regime:
            # Only take extreme mean reversion signals
            if fisher_extreme_oversold := fisher[i] < -2.5:
                new_signal = SIZE_RANGE * 0.5
            elif fisher_extreme_overbought := fisher[i] > 2.5:
                new_signal = -SIZE_RANGE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals