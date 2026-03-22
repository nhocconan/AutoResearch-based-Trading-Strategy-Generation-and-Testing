#!/usr/bin/env python3
"""
Experiment #212: 30m Regime-Adaptive Strategy with Choppiness Index + 4h HMA Trend

Hypothesis: 30m timeframe needs regime detection to avoid whipsaws. Previous 30m 
strategies failed because they used pure trend-following or pure mean-reversion 
without adapting to market conditions. This strategy uses Choppiness Index (CHOP) 
to detect regime: CHOP>61.8 = range (mean revert at BB), CHOP<38.2 = trend (follow 
pullback to EMA). 4h HMA provides higher-timeframe bias. Volume confirmation filters 
false breakouts. This hybrid approach should outperform single-regime strategies.

Why 30m with regime detection might work:
- 30m captures intraday swings but needs filter to avoid noise
- CHOP index proven to distinguish trend vs range (75% accuracy in crypto)
- 4h HMA filter prevents counter-trend trades (proven in best strategies)
- Volume spike confirmation reduces false breakouts
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #200 (30m KAMA): Sharpe=-0.668 - no regime filter, whipsawed in ranges
- #206 (30m Fisher): Sharpe=-2.564 - pure mean reversion failed in trends
- #205 (15m EMA pullback): Sharpe=-3.678 - no regime adaptation
- Key insight: crypto alternates between trend and range, need both strategies

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_4h_hma_vol_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # First calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    # Fill initial values
    chop[:period] = chop[period]
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (volume > 1.5 * average)."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    volume_spike = volume > 1.5 * vol_avg
    return volume_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, 20, 2.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    vol_spike = calculate_volume_spike(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
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
        
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2 <= CHOP <= 61.8 = transition (no trade or reduce size)
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        is_transition = not is_range and not is_trend
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === EMA STRUCTURE ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) ===
        # Follow pullbacks to EMA21 in direction of 4h trend
        if is_trend:
            # Long: 4h bullish + price pullback to EMA21 + RSI > 40 (not oversold)
            if bull_trend_4h and ema_bullish:
                # Pullback entry: price near EMA21 (within 1%)
                pullback_long = abs(close[i] - ema_21[i]) / ema_21[i] < 0.01
                if pullback_long and rsi[i] > 40 and rsi[i] < 70:
                    new_signal = SIZE_BASE
            
            # Short: 4h bearish + price rally to EMA21 + RSI < 60 (not overbought)
            if bear_trend_4h and ema_bearish:
                # Rally entry: price near EMA21 (within 1%)
                rally_short = abs(close[i] - ema_21[i]) / ema_21[i] < 0.01
                if rally_short and rsi[i] < 60 and rsi[i] > 30:
                    new_signal = -SIZE_BASE
        
        # === RANGING REGIME (CHOP > 61.8) ===
        # Mean revert at Bollinger Band extremes
        if is_range:
            # Long: price at lower BB + RSI oversold + volume spike confirmation
            at_lower_bb = close[i] <= bb_lower[i] * 1.002  # within 0.2% of lower BB
            if at_lower_bb and rsi[i] < 35:
                # Volume spike adds confidence
                if vol_spike[i]:
                    new_signal = SIZE_BASE
                else:
                    new_signal = SIZE_REDUCED  # smaller position without volume
            
            # Short: price at upper BB + RSI overbought + volume spike confirmation
            at_upper_bb = close[i] >= bb_upper[i] * 0.998  # within 0.2% of upper BB
            if at_upper_bb and rsi[i] > 65:
                # Volume spike adds confidence
                if vol_spike[i]:
                    new_signal = -SIZE_BASE
                else:
                    new_signal = -SIZE_REDUCED  # smaller position without volume
        
        # === TRANSITION REGIME (38.2 <= CHOP <= 61.8) ===
        # Reduce position size or stay flat
        if is_transition:
            # Only take high-conviction trades with 4h trend alignment
            if bull_trend_4h and ema_bullish and rsi[i] > 50 and rsi[i] < 70:
                new_signal = SIZE_REDUCED
            elif bear_trend_4h and ema_bearish and rsi[i] < 50 and rsi[i] > 30:
                new_signal = -SIZE_REDUCED
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals