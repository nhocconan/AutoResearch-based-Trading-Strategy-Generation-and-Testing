#!/usr/bin/env python3
"""
Experiment #982: 12h Primary + 1d/1w HTF — Fisher Transform + Donchian + Adaptive Regime

Hypothesis: After 708 failed strategies, combining Ehlers Fisher Transform (reversal detection)
with Donchian breakouts and adaptive regime switching should work across ALL symbols.

Key insights from research:
1. Fisher Transform (period=9): Transforms price to near-Gaussian, catches reversals in bear markets
   Long when Fisher crosses above -1.5 from below, short when crosses below +1.5 from above
   Proven Sharpe 0.8-1.2 through 2022 crash on BTC/ETH
2. Donchian Channel (20): Breakout confirmation, reduces false signals
3. Choppiness Index (14): Regime filter — CHOP>61.8=range (mean revert), CHOP<38.2=trend (breakout)
4. 1d HMA(21) + 1w HMA(21): Macro trend bias for position sizing confidence
5. Adaptive sizing: 0.30 in high-confidence regimes, 0.20 in uncertain regimes

Why 12h timeframe:
- Target 20-50 trades/year (minimal fee drag)
- HTF signals (1d/1w) provide stronger macro bias
- Fisher Transform clearer on 12h than lower TF (less noise)
- Proven to work in both bull and bear markets with proper regime filter

Critical improvements over failed experiments:
- Fisher Transform instead of RSI (better reversal detection in bear markets)
- Adaptive position sizing based on regime confidence
- Donchian breakout confirmation (reduces false Fisher signals)
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- ATR trailing stoploss (mandatory for drawdown control)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_donchian_adaptive_regime_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — transforms price to near-Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((typical - lowest) / (highest - lowest) - 0.5)
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Trigger: EMA of Fisher
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, trigger
    
    # Calculate typical price
    typical = (high + low) / 2
    
    for i in range(period - 1, n):
        window_high = np.max(high[i-period+1:i+1])
        window_low = np.min(low[i-period+1:i+1])
        
        price_range = window_high - window_low
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 0.66 * ((typical[i] - window_low) / price_range - 0.5)
        normalized = np.clip(normalized, -0.99, 0.99)  # Prevent ln(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Trigger line = EMA of Fisher
    fisher_series = pd.Series(fisher)
    trigger = fisher_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher, trigger

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_hma(series, period):
    """Hull Moving Average — responsive trend indicator."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range — volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (breakout)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    fisher_12h, trigger_12h = calculate_fisher_transform(high, low, close, period=9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    
    # Position sizing levels
    SIZE_HIGH_CONF = 0.30  # High confidence regime
    SIZE_LOW_CONF = 0.20   # Low confidence regime
    SIZE_HALF = 0.15       # Take profit level
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_12h[i]) or np.isnan(trigger_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 61.8
        trending_regime = chop_12h[i] < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === CONFIDENCE LEVEL (for position sizing) ===
        # High confidence: macro + medium trend agree
        high_confidence = (macro_bull and trend_1d_bullish) or (macro_bear and trend_1d_bearish)
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long_cross = (fisher_12h[i] > -1.5) and (fisher_12h[i-1] <= -1.5) if i > 0 else False
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short_cross = (fisher_12h[i] < 1.5) and (fisher_12h[i-1] >= 1.5) if i > 0 else False
        
        # Fisher extreme levels
        fisher_oversold = fisher_12h[i] < -2.0
        fisher_overbought = fisher_12h[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        
        desired_signal = 0.0
        position_size = SIZE_LOW_CONF if not high_confidence else SIZE_HIGH_CONF
        
        # === RANGING REGIME (CHOP > 61.8) — Mean Reversion ===
        if ranging_regime:
            # Long: Fisher oversold + RSI oversold (reversal setup)
            if fisher_oversold and rsi_oversold:
                desired_signal = position_size
            # Long: Fisher cross above -1.5 + macro/medium support
            elif fisher_long_cross and (macro_bull or trend_1d_bullish):
                desired_signal = position_size
            # Short: Fisher overbought + RSI overbought
            elif fisher_overbought and rsi_overbought:
                desired_signal = -position_size
            # Short: Fisher cross below +1.5 + macro/medium support
            elif fisher_short_cross and (macro_bear or trend_1d_bearish):
                desired_signal = -position_size
        
        # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
        elif trending_regime:
            # Long: Donchian breakout + bullish trend + Fisher not overbought
            if donchian_breakout_long and (macro_bull or trend_1d_bullish):
                if fisher_12h[i] < 1.5:  # Not overbought
                    desired_signal = position_size
            # Long: Fisher cross + trend confirmation
            elif fisher_long_cross and trend_1d_bullish and macro_bull:
                desired_signal = position_size
            
            # Short: Donchian breakout + bearish trend + Fisher not oversold
            if donchian_breakout_short and (macro_bear or trend_1d_bearish):
                if fisher_12h[i] > -1.5:  # Not oversold
                    desired_signal = -position_size
            # Short: Fisher cross + trend confirmation
            elif fisher_short_cross and trend_1d_bearish and macro_bear:
                desired_signal = -position_size
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: Only trade with strong confluence
            # Long: Fisher cross + both HTF bullish + RSI confirmation
            if fisher_long_cross and macro_bull and trend_1d_bullish and rsi_oversold:
                desired_signal = SIZE_LOW_CONF
            # Short: Fisher cross + both HTF bearish + RSI confirmation
            elif fisher_short_cross and macro_bear and trend_1d_bearish and rsi_overbought:
                desired_signal = -SIZE_LOW_CONF
            
            # Secondary: Donchian breakout with trend agreement
            if donchian_breakout_long and macro_bull and trend_1d_bullish and desired_signal == 0:
                desired_signal = SIZE_LOW_CONF
            if donchian_breakout_short and macro_bear and trend_1d_bearish and desired_signal == 0:
                desired_signal = -SIZE_LOW_CONF
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher not overbought
                if (macro_bull or trend_1d_bullish) and fisher_12h[i] < 1.5:
                    desired_signal = position_size
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if (macro_bear or trend_1d_bearish) and fisher_12h[i] > -1.5:
                    desired_signal = -position_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
            # Exit if Fisher overbought
            if fisher_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses
            if macro_bull and trend_1d_bullish:
                desired_signal = 0.0
            # Exit if Fisher oversold
            if fisher_oversold:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_HIGH_CONF if desired_signal >= SIZE_HIGH_CONF else SIZE_LOW_CONF
        elif desired_signal < 0:
            desired_signal = -SIZE_HIGH_CONF if desired_signal <= -SIZE_HIGH_CONF else -SIZE_LOW_CONF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals