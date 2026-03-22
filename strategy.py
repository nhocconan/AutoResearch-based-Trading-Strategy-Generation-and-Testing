#!/usr/bin/env python3
"""
Experiment #329: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Volatility Filter

Hypothesis: 4h timeframe with 1d trend filter captures medium-term crypto trends while avoiding
the whipsaw of lower timeframes. Key innovations:
1. 1d HMA(21) provides clean trend direction without excessive lag
2. 4h RSI(14) pullback entries (35-55 for longs, 45-65 for shorts) generate sufficient trades
3. ATR volatility filter reduces position size during extreme volatility (protects from 2022 crash)
4. Simple logic = more trades generated (avoiding the 0-trade failure mode)
5. Asymmetric bias: favor longs in bull regime, allow shorts only in strong bear regime
6. Target: 25-45 trades/year on 4h (appropriate frequency, low fee drag)

Why this might beat current best (Sharpe=0.424):
- 4h captures more trend moves than 12h/1d while avoiding 1h noise
- 1d HTF filter is cleaner than 4h/12h for major direction
- Simpler entry conditions = more trades (avoiding Sharpe=0.000 failure)
- Volatility scaling protects during crash periods (2022)
- Discrete signal levels (0.0, ±0.25, ±0.30) minimize fee churn

Position sizing: 0.25 base, 0.30 strong conviction, max 0.35
Stoploss: 2.5 * ATR trailing (tighter than 3.0 to protect capital)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_1d_asym_v1"
timeframe = "4h"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Reduces lag while maintaining smoothness.
    """
    n = period
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = wma(close_s, half_n)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bb(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_48 = calculate_hma(close, period=48)
    sma_50 = calculate_sma(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bb(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    MAX_SIGNAL = 0.35
    
    # Track state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]):
            signals[i] = 0.0
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA (favor longs)
        # Bear: price below 1d HMA (allow shorts)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        # Reduce size during extreme volatility (protects from crash)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.8
        vol_scale = 0.6 if high_vol else 1.0
        
        # === 4H LOCAL TREND ===
        # HMA crossover direction
        hma_bullish = hma_4h_21[i] > hma_4h_48[i]
        hma_bearish = hma_4h_21[i] < hma_4h_48[i]
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_4h_21[i]
        price_below_hma = close[i] < hma_4h_21[i]
        
        # Price relative to SMA50 (medium-term trend)
        price_above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False
        price_below_sma50 = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else False
        
        # === RSI SIGNALS (pullback entries) ===
        # Longs: RSI pullback to 35-55 in uptrend
        # Shorts: RSI pullback to 45-65 in downtrend
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # RSI momentum
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995
        price_at_bb_mid = abs(close[i] - bb_mid[i]) < (bb_upper[i] - bb_lower[i]) * 0.15
        
        # === ENTRY LOGIC (ASYMMETRIC + REGIME-AWARE) ===
        new_signal = 0.0
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: RSI pullback + HMA bullish + price above HMA
            if rsi_pullback_long and hma_bullish and price_above_hma:
                new_signal = LONG_BASE * vol_scale
            
            # Strong: RSI oversold + bull regime + above SMA50
            elif rsi_oversold and regime_bull and price_above_sma50:
                new_signal = LONG_STRONG * vol_scale
            
            # BB mean revert: price at lower band + RSI oversold + bull regime
            elif price_near_bb_lower and rsi_oversold and regime_bull:
                new_signal = LONG_BASE * vol_scale
            
            # HMA crossover + RSI rising
            elif hma_bullish and rsi_rising and rsi_14[i] > 40.0 and price_above_hma:
                new_signal = LONG_BASE * vol_scale
            
            # Extreme oversold bounce (any regime)
            elif rsi_extreme_oversold and rsi_rising:
                new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Primary: RSI pullback + HMA bearish + price below HMA
            if rsi_pullback_short and hma_bearish and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Strong: RSI overbought + bear regime + below SMA50
            elif rsi_overbought and regime_bear and price_below_sma50:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # BB mean revert: price at upper band + RSI overbought + bear regime
            elif price_near_bb_upper and rsi_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # HMA crossover + RSI falling
            elif hma_bearish and rsi_falling and rsi_14[i] < 60.0 and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Extreme overbought rejection (any regime)
            elif rsi_extreme_overbought and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === EXIT SIGNALS ===
        exit_signal = False
        
        if in_position and position_side != 0:
            # Long position exits
            if position_side > 0:
                # RSI overbought exit
                if rsi_extreme_overbought:
                    exit_signal = True
                # HMA turns bearish + price below
                elif hma_bearish and price_below_hma:
                    exit_signal = True
                # Regime turns strongly bearish
                elif regime_bear and price_below_sma50:
                    exit_signal = True
            
            # Short position exits
            if position_side < 0:
                # RSI oversold exit
                if rsi_extreme_oversold:
                    exit_signal = True
                # HMA turns bullish + price above
                elif hma_bullish and price_above_hma:
                    exit_signal = True
                # Regime turns strongly bullish
                elif regime_bull and price_above_sma50:
                    exit_signal = True
        
        if stoploss_triggered or exit_signal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0:
                if new_signal >= 0.28:
                    new_signal = min(LONG_STRONG * vol_scale, MAX_SIGNAL)
                else:
                    new_signal = min(LONG_BASE * vol_scale, MAX_SIGNAL)
            else:
                if new_signal <= -0.23:
                    new_signal = max(-SHORT_STRONG * vol_scale, -MAX_SIGNAL)
                else:
                    new_signal = max(-SHORT_BASE * vol_scale, -MAX_SIGNAL)
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals