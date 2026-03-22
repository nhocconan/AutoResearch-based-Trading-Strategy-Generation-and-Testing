#!/usr/bin/env python3
"""
Experiment #099: 1h Fisher Transform + 4h HMA Trend + Volatility Regime + RSI Confirmation

Hypothesis: 1h timeframe is ideal for mean-reversion-with-trend strategies.
Fisher Transform (Ehlers) excels at catching reversals in bear/range markets.
4h HMA provides stable trend bias without excessive lag.
Volatility filter (ATR ratio) avoids entering during panic spikes.
RSI confirmation ensures momentum alignment.

Why this might work (learning from failures):
- #087, #093 (1h strategies) failed with negative Sharpe - too much noise
- #090, #096 (1d strategies) had positive Sharpe but low trade count
- Key insight: 1h needs MEAN REVERSION entries with TREND filter (not pure trend)
- Fisher Transform catches reversals better than RSI extremes
- Asymmetric logic: more aggressive longs in bull, cautious shorts in bear
- Looser entry conditions to ensure ≥10 trades per symbol

Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.20 base, 0.30 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_vol_regime_rsi_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Calculate typical price
    hl2 = (high + low) / 2
    typical = (2 * hl2 + close) / 3
    
    # Normalize price to -1 to +1 range
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 0.001, 0.001, range_val)
    
    normalized = 2 * (typical - lowest) / range_val - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Fisher signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Fisher Transform for reversal signals
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    
    # ATR ratio for volatility spike detection
    atr_ratio = atr_7 / atr_30
    atr_ratio = np.where(np.isnan(atr_ratio) | (atr_30 == 0), 1.0, atr_ratio)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # EMA alignment on 1h
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === VOLATILITY REGIME FILTER ===
        # High vol (atr_ratio > 1.5) = panic, wait for reversion
        # Normal vol (atr_ratio < 1.2) = good for entries
        # Extreme vol (atr_ratio > 2.0) = avoid new entries
        vol_normal = atr_ratio[i] < 1.5
        vol_spike = atr_ratio[i] > 1.8
        vol_extreme = atr_ratio[i] > 2.5
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher in extreme zones (mean reversion setup)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # === BOLLINGER BAND POSITION ===
        price_near_lower = close[i] <= bb_lower[i] * 1.005  # within 0.5% of lower band
        price_near_upper = close[i] >= bb_upper[i] * 0.995  # within 0.5% of upper band
        price_below_mid = close[i] < bb_mid[i]
        price_above_mid = close[i] > bb_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (mean reversion with trend filter) ===
        # Path 1: Fisher oversold cross + 4h bullish + normal vol (strong long)
        if fisher_long_cross and bull_trend_4h and vol_normal:
            new_signal = SIZE_STRONG
        
        # Path 2: Fisher oversold + price at BB lower + 4h bullish (strong long)
        if new_signal == 0.0 and fisher_oversold and price_near_lower and bull_trend_4h:
            if vol_normal or rsi_oversold:
                new_signal = SIZE_STRONG
        
        # Path 3: RSI oversold + price at BB lower + 4h bullish (mean reversion long)
        if new_signal == 0.0 and rsi_oversold and price_near_lower and bull_trend_4h:
            if vol_normal:
                new_signal = SIZE_BASE
        
        # Path 4: Fisher oversold + 4h bullish (simpler, ensures trades)
        if new_signal == 0.0 and fisher_oversold and bull_trend_4h:
            if ema_bullish or rsi_neutral:
                new_signal = SIZE_BASE
        
        # Path 5: Price at BB lower + 4h bullish + normal vol (fallback)
        if new_signal == 0.0 and price_near_lower and bull_trend_4h and vol_normal:
            if not vol_extreme:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (mean reversion with trend filter) ===
        # Path 1: Fisher overbought cross + 4h bearish + normal vol (strong short)
        if fisher_short_cross and bear_trend_4h and vol_normal:
            new_signal = -SIZE_STRONG
        
        # Path 2: Fisher overbought + price at BB upper + 4h bearish (strong short)
        if new_signal == 0.0 and fisher_overbought and price_near_upper and bear_trend_4h:
            if vol_normal or rsi_overbought:
                new_signal = -SIZE_STRONG
        
        # Path 3: RSI overbought + price at BB upper + 4h bearish (mean reversion short)
        if new_signal == 0.0 and rsi_overbought and price_near_upper and bear_trend_4h:
            if vol_normal:
                new_signal = -SIZE_BASE
        
        # Path 4: Fisher overbought + 4h bearish (simpler, ensures trades)
        if new_signal == 0.0 and fisher_overbought and bear_trend_4h:
            if ema_bearish or rsi_neutral:
                new_signal = -SIZE_BASE
        
        # Path 5: Price at BB upper + 4h bearish + normal vol (fallback)
        if new_signal == 0.0 and price_near_upper and bear_trend_4h and vol_normal:
            if not vol_extreme:
                new_signal = -SIZE_BASE
        
        # === VOLATILITY SPIKE EXIT ===
        # If in position and vol spikes to extreme, reduce/exit
        if in_position and vol_extreme:
            # Don't add to position during panic, but don't exit immediately
            if new_signal != 0.0:
                new_signal = new_signal * 0.5  # Reduce size during panic
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
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