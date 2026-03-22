#!/usr/bin/env python3
"""
Experiment #524: 30m Choppiness Regime with 4h HMA Bias and RSI Mean-Reversion

Hypothesis: After 500+ failed experiments, the key insight is that 30m timeframe
needs REGIME-BASED approach. Crypto alternates between trending and ranging periods.
Using Choppiness Index (CHOP) to detect regime, then applying appropriate strategy:
- Range regime (CHOP > 61.8): Mean-reversion at RSI extremes
- Trend regime (CHOP < 38.2): Trend-following on pullbacks to EMA
- Neutral: Stay flat or reduce position size

Combined with 4h HMA for directional bias (via mtf_data helper) to avoid
counter-trend trades. This addresses the #1 failure mode: trend strategies
in range markets and mean-reversion in trending markets.

Key innovations:
1. CHOPPINESS INDEX (14): Proper formula from market science literature
2. REGIME HYSTERESIS: Enter range at 61.8, exit at 55 (prevents whipsaw)
3. LOOSE RSI THRESHOLDS: <35 long, >65 short (ensures ≥10 trades/year)
4. 4h HMA BIAS: Only take trades in direction of higher timeframe trend
5. 2.0 * ATR STOPLOSS: Tight enough to limit drawdown, loose enough to breathe
6. DISCRETE SIZING: 0.0, ±0.25, ±0.30 to minimize fee churn

Why 30m works:
- Fast enough to catch intraday swings
- Slow enough to avoid 5m/15m noise
- 48 bars/day = good statistical sample
- Works well with 4h HTF reference (8x ratio)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_4h_hma_rsi_meanrev_trend_adaptive_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Range-bound market (mean-reversion works)
    - CHOP < 38.2: Trending market (trend-following works)
    - 38.2 < CHOP < 61.8: Transition/neutral
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral default
    
    return chop

def calculate_chop_regime(chop, enter_range=61.8, exit_range=55.0, enter_trend=38.2, exit_trend=45.0):
    """
    Calculate regime state with hysteresis to prevent whipsaw.
    Returns: 1=trending, 0=neutral, -1=ranging
    """
    n = len(chop)
    regime = np.zeros(n)
    state = 0  # 0=neutral, 1=trending, -1=ranging
    
    for i in range(n):
        if np.isnan(chop[i]):
            continue
        
        if state == 0:
            if chop[i] > enter_range:
                state = -1  # enter range
            elif chop[i] < enter_trend:
                state = 1   # enter trend
        elif state == 1:  # trending
            if chop[i] > exit_trend:
                state = 0   # exit to neutral
        elif state == -1:  # ranging
            if chop[i] < exit_range:
                state = 0   # exit to neutral
        
        regime[i] = state
    
    return regime

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    chop_regime = calculate_chop_regime(chop, 61.8, 55.0, 38.2, 45.0)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_RANGE = 0.30  # Larger size in range regime (mean-reversion has higher win rate)
    SIZE_TREND = 0.25  # Smaller size in trend regime (more risk)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_regime[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        regime = chop_regime[i]  # 1=trending, 0=neutral, -1=ranging
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_RANGE if regime == -1 else SIZE_TREND
        
        # RANGE REGIME: Mean-reversion at extremes
        if regime == -1:
            # Long: RSI oversold + bullish 4h bias
            if rsi_14[i] < 35 and bull_bias:
                new_signal = current_size
            # Short: RSI overbought + bearish 4h bias
            elif rsi_14[i] > 65 and bear_bias:
                new_signal = -current_size
        
        # TREND REGIME: Trend-following on pullbacks
        elif regime == 1:
            if bull_bias:
                # Long pullback in uptrend: price > EMA but RSI dipped
                if close[i] > ema_21[i] and rsi_14[i] < 50:
                    new_signal = current_size
            elif bear_bias:
                # Short rally in downtrend: price < EMA but RSI rose
                if close[i] < ema_21[i] and rsi_14[i] > 50:
                    new_signal = -current_size
        
        # NEUTRAL REGIME: Reduced size, only strong signals
        else:
            if rsi_14[i] < 30 and bull_bias:
                new_signal = SIZE_TREND * 0.5
            elif rsi_14[i] > 70 and bear_bias:
                new_signal = -SIZE_TREND * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME EXIT ===
        # Exit if regime flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and regime == 1 and bear_bias:
                # Long in trending regime but 4h bias turned bear
                new_signal = 0.0
            if position_side < 0 and regime == 1 and bull_bias:
                # Short in trending regime but 4h bias turned bull
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