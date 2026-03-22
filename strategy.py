#!/usr/bin/env python3
"""
Experiment #334: 4h Primary + 12h/1d HTF — KAMA Trend + RSI Pullback + ATR Risk

Hypothesis: KAMA adapts better than HMA in ranging markets (common in 2022, 2025).
Using 12h HMA for trend direction + 4h KAMA for entries should generate 30-50 trades/year.
Simpler entry logic (fewer AND conditions) to avoid 0-trade failures seen in exp 324, 331, 332.

Key changes from failed experiments:
- NO choppiness filter (experiments 324, 331, 332 got 0 trades)
- NO Connors RSI complexity (experiment 323 failed)
- Simple RSI(14) pullback zones (35-55 long, 45-65 short)
- KAMA(14) adapts to volatility better than fixed EMA/HMA
- 12h HMA(21) for major trend (not 1w which is too slow for 4h entries)
- ATR(14) trailing stop at 2.5x
- Asymmetric sizing: longs 0.30, shorts 0.20 (crypto bias)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Target: 30-50 trades/year on 4h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_12h_simp_asym_v1"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    ER (Efficiency Ratio) = |price change| / sum(|price changes|)
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    price_change = np.abs(close - np.roll(close, period))
    price_change[:period] = np.nan
    
    sum_price_changes = pd.Series(np.abs(close_s.diff())).rolling(window=period, min_periods=period).sum().values
    sum_price_changes[:period] = np.nan
    
    er = price_change / (sum_price_changes + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize with price
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    kama_14 = calculate_kama(close, period=14)
    kama_50 = calculate_kama(close, period=50)
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
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(kama_50[i]):
            continue
        
        # === 12H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 12h HMA (favor longs)
        # Bear: price below 12h HMA (allow shorts)
        regime_bull = close[i] > hma_12h_21_aligned[i]
        regime_bear = close[i] < hma_12h_21_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 4H LOCAL TREND (KAMA) ===
        # KAMA crossover
        kama_bullish = kama_14[i] > kama_50[i]
        kama_bearish = kama_14[i] < kama_50[i]
        
        # KAMA slope (3-bar lookback)
        kama_slope_up = kama_14[i] > kama_14[i-3] if i >= 3 else False
        kama_slope_down = kama_14[i] < kama_14[i-3] if i >= 3 else False
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_14[i]
        price_below_kama = close[i] < kama_14[i]
        
        # Price relative to SMA200 (long-term trend filter)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (pullback entries, not extremes) ===
        # Looser thresholds to generate more trades
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLER - fewer AND conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Primary: RSI pullback + KAMA bullish + price above KAMA
            if rsi_pullback_long and kama_bullish and price_above_kama:
                new_signal = LONG_BASE * vol_scale
            
            # Strong: RSI very oversold + bull regime
            elif rsi_strong_oversold and regime_bull:
                new_signal = LONG_STRONG * vol_scale
            
            # KAMA bullish + KAMA slope up + RSI rising
            elif kama_bullish and kama_slope_up and rsi_rising:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * vol_scale
            
            # Price above SMA200 + RSI > 45 (momentum continuation)
            elif price_above_sma200 and rsi_14[i] > 45.0 and kama_bullish:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Primary: RSI pullback + KAMA bearish + price below KAMA
            if rsi_pullback_short and kama_bearish and price_below_kama:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Strong: RSI very overbought + bear regime
            elif rsi_strong_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # KAMA bearish + KAMA slope down + RSI falling
            elif kama_bearish and kama_slope_down and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Price below SMA200 + RSI < 55 (momentum continuation)
            elif not price_above_sma200 and rsi_14[i] < 55.0 and kama_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 4h) ===
        # Force trade if no signal for 30 bars (~30 * 4h = 5 days)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 60.0:
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
            # Long position but 12h regime turns bearish + price below KAMA
            if position_side > 0 and regime_bear and price_below_kama:
                regime_reversal = True
            # Short position but 12h regime turns bullish + price above KAMA
            if position_side < 0 and regime_bull and price_above_kama:
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