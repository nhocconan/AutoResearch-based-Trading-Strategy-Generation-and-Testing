#!/usr/bin/env python3
"""
Experiment #063: 1h Fisher Transform + 4h HMA Trend + Volatility Regime
Hypothesis: 1h timeframe balances signal frequency and noise. Fisher Transform catches 
reversals better than RSI in bear/range markets (proven in Ehlers literature). 
4h HMA provides trend bias without excessive lag. Volatility regime adjusts position size.
Why this might work: Fisher Transform normalizes price to Gaussian distribution, making 
extreme readings (-2/+2) reliable reversal signals. Combined with HTF trend filter, this 
avoids counter-trend trades while catching pullbacks in trends and reversals in ranges.
Key improvement: Looser entry thresholds to ensure 10+ trades per symbol (learned from failures).
Position sizing: 0.20-0.30 discrete levels, stoploss at 2.5*ATR trailing.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_vol_regime_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    # Calculate median price
    median = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        normalized = (median[i] - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_val
            trigger[i] = fisher_val
        else:
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def calculate_volatility_ratio(atr_short, atr_long):
    """Calculate ratio of short-term to long-term ATR for vol regime."""
    return atr_short / (atr_long + 1e-10)

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
    rsi_7 = calculate_rsi(close, 7)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    # Fisher Transform for reversals
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    
    # Volatility ratio for regime
    vol_ratio = calculate_volatility_ratio(atr_7, atr_30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.20
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # EMA alignment on 1h
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === VOLATILITY REGIME ===
        high_vol = vol_ratio[i] > 1.5
        low_vol = vol_ratio[i] < 0.8
        normal_vol = 0.8 <= vol_ratio[i] <= 1.5
        
        # Adjust size based on vol
        if high_vol:
            current_size = SIZE_WEAK  # Reduce size in high vol
        elif low_vol:
            current_size = SIZE_STRONG  # Increase size in low vol
        else:
            current_size = SIZE_BASE
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # Extreme Fisher readings (stronger signals)
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.01 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.99 if not np.isnan(bb_upper[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Fisher reversal + bull trend (primary signal)
        if bull_trend_4h:
            if fisher_long_cross:
                new_signal = current_size
            elif fisher_extreme_long and rsi_oversold:
                new_signal = current_size
        
        # Path 2: Fisher + EMA alignment
        if bull_trend_4h and ema_bullish:
            if fisher[i] < -1.0 and rsi[i] < 50:
                new_signal = SIZE_BASE
        
        # Path 3: Mean reversion in range (price above SMA200)
        if above_sma200 and not bear_trend_4h:
            if rsi_oversold and near_bb_lower:
                new_signal = SIZE_WEAK
            elif fisher_extreme_long:
                new_signal = SIZE_WEAK
        
        # Path 4: Pullback entry in uptrend
        if bull_trend_4h and ema_bullish:
            if close[i] < ema_21[i] * 0.995 and rsi[i] < 45:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Fisher reversal + bear trend (primary signal)
        if bear_trend_4h:
            if fisher_short_cross:
                new_signal = -current_size
            elif fisher_extreme_short and rsi_overbought:
                new_signal = -current_size
        
        # Path 2: Fisher + EMA alignment
        if bear_trend_4h and ema_bearish:
            if fisher[i] > 1.0 and rsi[i] > 50:
                new_signal = -SIZE_BASE
        
        # Path 3: Mean reversion in range (price below SMA200)
        if below_sma200 and not bull_trend_4h:
            if rsi_overbought and near_bb_upper:
                new_signal = -SIZE_WEAK
            elif fisher_extreme_short:
                new_signal = -SIZE_WEAK
        
        # Path 4: Pullback entry in downtrend
        if bear_trend_4h and ema_bearish:
            if close[i] > ema_21[i] * 1.005 and rsi[i] > 55:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Exit
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals