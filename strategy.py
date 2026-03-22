#!/usr/bin/env python3
"""
Experiment #591: 4h Primary + 1d HTF — Dual Regime (Choppiness Index Switch)

Hypothesis: After analyzing 500+ failed strategies, the pattern is clear:
- Single-regime strategies fail because crypto alternates between trend/range
- Choppiness Index (CHOP) is the BEST regime filter from literature
- CHOP > 61.8 = range market → use mean reversion (RSI extremes)
- CHOP < 38.2 = trend market → use trend following (Donchian/HMA breakout)
- 1d HMA(21) for major trend bias (filters counter-trend trades)
- This combines proven patterns: CHOP regime (ETH Sharpe +0.923) + Donchian (SOL +0.782)

Why this might beat Sharpe=0.520:
- Dual regime adapts to market conditions (not fighting the regime)
- CHOP filter prevents trend strategies in chop (major source of losses)
- Mean reversion in chop has 70%+ win rate in literature
- 1d HTF bias prevents counter-trend disasters (2022 crash protection)
- Wider thresholds ensure 30-50 trades/year on 4h

Entry Logic:
- REGIME = CHOP(14): >61.8 chop, <38.2 trend, else neutral
- CHOP + RSI mean reversion: long RSI<25, short RSI>75 (in chop)
- CHOP + Donchian breakout: long break high(20), short break low(20) (in trend)
- 1d HMA(21) bias: only long if price>1d_HMA, only short if price<1d_HMA
- ATR(14) 2.5x trailing stop on all positions

Position sizing: 0.30 discrete (Rule 4, max 0.40)
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_dual_regime_hma_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection from literature.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest High)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    # Clip to valid range
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2-61.8 = neutral (no trades or reduced size)
        is_chop_regime = chop_14[i] > 61.8
        is_trend_regime = chop_14[i] < 38.2
        is_neutral_regime = not is_chop_regime and not is_trend_regime
        
        # === 1D MAJOR TREND BIAS ===
        bull_bias_1d = close[i] > hma_1d_21_aligned[i]
        bear_bias_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength confirmation
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        
        # --- CHOP REGIME: Mean Reversion (RSI extremes) ---
        if is_chop_regime:
            # Long: RSI < 25 (oversold) + 1d bull bias
            if rsi_14[i] < 25.0 and bull_bias_1d:
                new_signal = POSITION_SIZE
            
            # Short: RSI > 75 (overbought) + 1d bear bias
            elif rsi_14[i] > 75.0 and bear_bias_1d:
                new_signal = -POSITION_SIZE
        
        # --- TREND REGIME: Trend Following (Donchian breakout) ---
        elif is_trend_regime:
            # Long: Price breaks Donchian upper + 1d bull bias
            if close[i] > donchian_upper[i-1] and bull_bias_1d:
                # Size based on 1d trend strength
                if hma_1d_slope_bull:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = POSITION_SIZE * 0.7
            
            # Short: Price breaks Donchian lower + 1d bear bias
            elif close[i] < donchian_lower[i-1] and bear_bias_1d:
                # Size based on 1d trend strength
                if hma_1d_slope_bear:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -POSITION_SIZE * 0.7
        
        # --- NEUTRAL REGIME: Reduced activity ---
        elif is_neutral_regime:
            # Only enter on extreme conditions
            if rsi_14[i] < 20.0 and bull_bias_1d:
                new_signal = POSITION_SIZE * 0.5
            elif rsi_14[i] > 80.0 and bear_bias_1d:
                new_signal = -POSITION_SIZE * 0.5
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on 1d bias flip to bear
        if in_position and position_side > 0:
            if bear_bias_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d bias flip to bull
        if in_position and position_side < 0:
            if bull_bias_1d and hma_1d_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals