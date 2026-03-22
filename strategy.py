#!/usr/bin/env python3
"""
Experiment #354: 1d Regime-Adaptive BB/RSI with ADX Filter

Hypothesis: After 352 failed experiments, the clearest pattern is OVER-FILTERING.
Strategies with 4-5 conflicting conditions generate 0 trades or miss major moves.

For DAILY timeframe specifically:
1. FEWER filters = more trades = better statistics
2. Regime detection (ADX) should SIMPLIFY not complicate entry logic
3. 200-SMA bias is essential for BTC/ETH (asymmetric response to bull/bear)
4. BB + RSI mean reversion works in range; Donchian breakout works in trend

Strategy Logic:
- REGIME: ADX < 25 = range (BB mean revert), ADX >= 25 = trend (Donchian breakout)
- BIAS: Price > SMA200 = bull (prefer long), Price < SMA200 = bear (prefer short)
- LONG: BB lower touch + RSI < 45 (range) OR Donchian breakout (trend, bull only)
- SHORT: BB upper touch + RSI > 55 (range) OR Donchian breakout (trend, bear only)
- STOPLOSS: 2.5 * ATR trailing
- SIZE: 0.30 discrete

Why this should work on 1d:
- Daily candles filter noise (unlike 15m/1h that failed repeatedly)
- ADX regime split prevents whipsaw (enter mean-reversion only in chop)
- SMA200 bias prevents counter-trend disasters (2022 crash lesson)
- Relaxed RSI thresholds (45/55 not 30/70) = more trades

Timeframe: 1d (REQUIRED)
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd

name = "mtf_1d_regime_adaptive_bb_rsi_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 10:
        return adx
    
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
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete (Rule 4)
    SIZE = 0.30
    
    # Calculate all indicators (vectorized before loop - Rule 8)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_mult = 2.0
    bb_mid = pd.Series(close).rolling(bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + bb_std * bb_mult
    bb_lower = bb_mid - bb_std * bb_mult
    
    # Donchian Channels (20-period breakout)
    donchian_high = pd.Series(high).rolling(20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(20, min_periods=20).min().values
    
    # SMA 200 for trend bias
    sma_200 = pd.Series(close).rolling(200, min_periods=200).mean().values
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Start from 250 to ensure all indicators ready (200 SMA + buffer)
    for i in range(250, n):
        # Skip invalid data
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(donchian_high[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # ADX < 25 = ranging market (mean reversion works)
        # ADX >= 25 = trending market (breakout works)
        is_ranging = adx[i] < 25
        is_trending = adx[i] >= 25
        
        # === TREND BIAS ===
        # Price above SMA200 = bull market (prefer long)
        # Price below SMA200 = bear market (prefer short)
        bull_bias = close[i] > sma_200[i]
        bear_bias = close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === MEAN REVERSION ENTRIES (Ranging Market) ===
        if is_ranging:
            # Long: Price at BB lower + RSI oversold
            # Relaxed thresholds for more trades (RSI < 45 not 30)
            if close[i] <= bb_lower[i] * 1.002 and rsi[i] < 45:
                # In bull market: take all mean-reversion longs
                # In bear market: only extreme oversold (RSI < 30)
                if bull_bias or rsi[i] < 30:
                    new_signal = SIZE
            
            # Short: Price at BB upper + RSI overbought
            if close[i] >= bb_upper[i] * 0.998 and rsi[i] > 55:
                # In bear market: take all mean-reversion shorts
                # In bull market: only extreme overbought (RSI > 70)
                if bear_bias or rsi[i] > 70:
                    new_signal = -SIZE
        
        # === TREND FOLLOWING ENTRIES (Trending Market) ===
        if is_trending:
            # Long breakout: Only in bull market
            if bull_bias and close[i] > donchian_high[i-1]:
                new_signal = SIZE
            
            # Short breakout: Only in bear market
            if bear_bias and close[i] < donchian_low[i-1]:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Long: update highest close, stop if price drops 2.5*ATR
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Short: update lowest close, stop if price rises 2.5*ATR
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND BIAS REVERSAL EXIT ===
        # Exit long if market turns bearish (price < SMA200)
        if in_position and position_side > 0 and bear_bias:
            new_signal = 0.0
        
        # Exit short if market turns bullish (price > SMA200)
        if in_position and position_side < 0 and bull_bias:
            new_signal = 0.0
        
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