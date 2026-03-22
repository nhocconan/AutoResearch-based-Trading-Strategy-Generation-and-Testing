#!/usr/bin/env python3
"""
Experiment #137: 12h Regime-Adaptive Strategy with 1d HMA + Choppiness Index + RSI Mean Reversion

Hypothesis: The 12h timeframe needs regime-adaptive logic to handle both trending and ranging markets.
Building on lessons from failed experiments:
- Pure trend following fails in 2022 crash and 2025 bear market
- Mean reversion alone fails in strong trends
- Solution: Use Choppiness Index (CHOP) to detect regime, then apply appropriate logic

Strategy components:
1. 1d HMA(21) = HTF trend bias (call get_htf_data ONCE before loop)
2. Choppiness Index(14) = regime detector (<38.2 trend, >61.8 range)
3. RSI(7) = entry trigger (extremes for mean reversion, pullback for trend)
4. ATR(14) = volatility measure for stoploss at 2.5*ATR
5. Regime-adaptive entry logic:
   - Trend regime (CHOP<38.2): enter on RSI pullback in 1d trend direction
   - Range regime (CHOP>61.8): mean revert at RSI extremes (<25 long, >75 short)
   - Neutral regime: stay flat or reduce position

Why this might beat Sharpe=0.478 baseline:
- Regime detection avoids whipsaw in choppy markets (2022 bottom, 2025 range)
- 12h timeframe = fewer false signals than 4h
- RSI(7) more responsive than RSI(14) for 12h bars
- Discrete position sizing (0.25/0.35) minimizes fee churn
- ATR stoploss protects from catastrophic moves

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (get_htf_data called ONCE before loop)
Position sizing: 0.25 base, 0.35 strong signals, discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_rsi_1d_hma_atr_v1"
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

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.zeros(len(close))
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging/choppy market (mean reversion likely)
    CHOP < 38.2 = trending market (trend following likely)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    # Calculate ATR for each bar (true range)
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
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Faster RSI for 12h
    chop = calculate_choppiness_index(high, low, close, 14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        # CHOP > 61.8 = range (mean reversion)
        # CHOP < 38.2 = trend (trend following)
        # 38.2 <= CHOP <= 61.8 = neutral (reduce exposure)
        regime_range = chop[i] > 61.8
        regime_trend = chop[i] < 38.2
        regime_neutral = not regime_range and not regime_trend
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        rsi_pullback_long = 35 < rsi[i] < 50  # Pullback in uptrend
        rsi_pullback_short = 50 < rsi[i] < 65  # Pullback in downtrend
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # Near or below lower BB
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # Near or above upper BB
        
        new_signal = 0.0
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        
        # TREND REGIME: Follow 1d trend on RSI pullbacks
        if regime_trend:
            # Long: 1d bullish + RSI pullback + not overbought
            if bull_trend_1d and rsi_pullback_long and not rsi_overbought:
                new_signal = SIZE_BASE
            # Strong long: add BB support
            elif bull_trend_1d and rsi_pullback_long and near_bb_lower:
                new_signal = SIZE_STRONG
            
            # Short: 1d bearish + RSI pullback + not oversold
            if bear_trend_1d and rsi_pullback_short and not rsi_oversold:
                new_signal = -SIZE_BASE
            # Strong short: add BB resistance
            elif bear_trend_1d and rsi_pullback_short and near_bb_upper:
                new_signal = -SIZE_STRONG
        
        # RANGE REGIME: Mean reversion at extremes
        elif regime_range:
            # Long: RSI oversold + near lower BB
            if rsi_oversold or near_bb_lower:
                new_signal = SIZE_BASE
            # Strong long: both conditions
            if rsi_oversold and near_bb_lower:
                new_signal = SIZE_STRONG
            
            # Short: RSI overbought + near upper BB
            if rsi_overbought or near_bb_upper:
                new_signal = -SIZE_BASE
            # Strong short: both conditions
            if rsi_overbought and near_bb_upper:
                new_signal = -SIZE_STRONG
        
        # NEUTRAL REGIME: Reduce exposure, only take strongest signals
        elif regime_neutral:
            # Only enter on extreme RSI with 1d trend confirmation
            if bull_trend_1d and rsi_oversold:
                new_signal = SIZE_BASE * 0.5  # Half size in neutral
            elif bear_trend_1d and rsi_overbought:
                new_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals