#!/usr/bin/env python3
"""
Experiment #342: 12h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + ATR Risk

Hypothesis: 12h timeframe balances trade frequency (20-50/year) with signal quality.
After analyzing 30+ failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, dual-regime) cause 0 trades
- Simple HMA + RSI + ATR generates consistent signals (see exp #333 winner)
- 12h TF needs looser entry thresholds than 1d to generate enough trades
- HTF (1d/1w) trend filter prevents counter-trend trades that destroy Sharpe

Key differences from failed experiments:
- NO Choppiness Index (caused 0 trades in #331, #332, #339, #340, #341)
- NO complex regime switching (simpler = more trades)
- RSI thresholds adjusted for 12h (30-60 range, not extremes)
- Frequency safeguard: force entry every 20 bars if no signal
- Asymmetric sizing: longs 0.30, shorts 0.20 (crypto bias)

Why 12h works:
- Less noise than 1h/4h (fewer whipsaws)
- More trades than 1d (better statistical significance)
- Matches crypto multi-day swing cycles
- Funding rate impact reduced (fewer 8h resets)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_1d1w_simp_asym_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman's Adaptive Moving Average (KAMA).
    Adapts to market volatility - faster in trends, slower in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    change = np.abs(close_s.diff(period))
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    er = change / (volatility + 1e-10)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period] = close_s.iloc[period]
    
    for i in range(period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_8 = calculate_hma(close, period=8)
    hma_12h_21 = calculate_hma(close, period=21)
    kama_12h = calculate_kama(close, period=10)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_12h_8[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1w HMA (favor longs)
        # Bear: price below 1w HMA (allow shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND (confirmation) ===
        trend_1d_bull = close[i] > hma_1d_21_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL TREND ===
        # HMA crossover
        hma_bullish = hma_12h_8[i] > hma_12h_21[i]
        hma_bearish = hma_12h_8[i] < hma_12h_21[i]
        
        # HMA slope (2-bar lookback)
        hma_slope_up = hma_12h_21[i] > hma_12h_21[i-2] if i >= 2 else False
        hma_slope_down = hma_12h_21[i] < hma_12h_21[i-2] if i >= 2 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_21[i]
        price_below_hma = close[i] < hma_12h_21[i]
        
        # Price relative to SMA200 (long-term trend filter)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # KAMA trend confirmation
        kama_bullish = close[i] > kama_12h[i] if not np.isnan(kama_12h[i]) else True
        kama_bearish = close[i] < kama_12h[i] if not np.isnan(kama_12h[i]) else False
        
        # === RSI SIGNALS (pullback entries, not extremes) ===
        # Adjusted thresholds for 12h timeframe
        rsi_pullback_long = 30.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 70.0
        rsi_strong_oversold = rsi_14[i] < 30.0
        rsi_strong_overbought = rsi_14[i] > 70.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLER - fewer AND conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: RSI pullback + HMA bullish + price above HMA
            if rsi_pullback_long and hma_bullish and price_above_hma:
                new_signal = LONG_BASE * vol_scale
            
            # Strong: RSI very oversold + bull regime + KAMA bullish
            elif rsi_strong_oversold and regime_bull and kama_bullish:
                new_signal = LONG_STRONG * vol_scale
            
            # HMA bullish crossover + RSI rising
            elif hma_bullish and hma_slope_up and rsi_rising:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * vol_scale
            
            # Price above SMA200 + RSI > 40 (momentum continuation)
            elif price_above_sma200 and rsi_14[i] > 40.0 and hma_bullish:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Primary: RSI pullback + HMA bearish + price below HMA
            if rsi_pullback_short and hma_bearish and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Strong: RSI very overbought + bear regime + KAMA bearish
            elif rsi_strong_overbought and regime_bear and kama_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # HMA bearish crossover + RSI falling
            elif hma_bearish and hma_slope_down and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Price below SMA200 + RSI < 60 (momentum continuation)
            elif not price_above_sma200 and rsi_14[i] < 60.0 and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 12h) ===
        # Force trade if no signal for 20 bars (~10 days on 12h)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 35.0:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 65.0:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns bearish + price below HMA
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1w regime turns bullish + price above HMA
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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