#!/usr/bin/env python3
"""
Experiment #290: 30m Volatility Spike Mean Reversion with 4h HMA Bias

Hypothesis: After 289 experiments, trend-following strategies consistently fail
in bear/range markets (2022 crash, 2025 bear). This strategy uses VOLATILITY
SPIKE MEAN REVERSION which has shown Sharpe 0.8-1.5 through crashes:

1. ATR Ratio Spike: ATR(7)/ATR(30) > 1.8 signals panic/extreme volatility
2. Bollinger Band Extreme: Price < BB_lower(20, 2.0) for long, > BB_upper for short
3. 4h HMA(21) Bias: Only trade mean reversion in direction of HTF trend
4. Volume Confirmation: Spike volume > 1.5x average confirms panic/reversal
5. Exit on Vol Normalization: ATR ratio < 1.2 signals vol crush = exit
6. Stoploss: 2.5 * ATR(14) trailing stop

Why this might work when others failed:
- Mean reversion works in bear/range markets (unlike pure trend)
- Vol spike captures panic bottoms (2022) and FOMO tops
- 4h HMA bias prevents counter-trend mean reversion (critical filter)
- 30m timeframe = frequent enough for >=10 trades, not too noisy
- Discrete position sizing (0.25/0.30) minimizes fee churn

Key differences from failed strategies:
- NO RSI pullback (failed #284, #285)
- NO Supertrend (failed #284, #289)
- NO pure KAMA trend (failed #278, #281, #283)
- Uses VOLATILITY not momentum for entry signal
- Mean reversion WITH trend filter (asymmetric)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete, max 0.35
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_spike_meanrev_4h_hma_bb_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.values

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
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    vol_sma = calculate_volume_sma(volume, 20)
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_HIGH = 0.30  # Higher size on strong signals
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss and exit logic
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    vol_spike_active = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR ratio > 1.8 signals panic/extreme volatility (entry trigger)
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 0
        vol_spike = atr_ratio > 1.8
        
        # Vol normalization (exit signal)
        vol_normal = atr_ratio < 1.2
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = directional bias (asymmetric mean reversion)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === BOLLINGER BAND EXTREMES ===
        # Price at BB extreme during vol spike = mean reversion opportunity
        price_at_bb_lower = close[i] < bb_lower[i] * 1.002  # slight buffer
        price_at_bb_upper = close[i] > bb_upper[i] * 0.998  # slight buffer
        
        # === Z-SCORE CONFIRMATION ===
        # Extreme Z-score confirms oversold/overbought
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # === VOLUME CONFIRMATION ===
        # Spike volume confirms panic/reversal
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # === DETERMINE POSITION SIZE ===
        # Higher size when multiple confirmations align
        if vol_spike and volume_spike:
            position_size = SIZE_HIGH
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Vol spike + price at BB lower + 4h bull trend + Z-score oversold
        # Asymmetric: only long mean reversion when 4h trend is bullish
        long_conditions = (
            vol_spike and  # Volatility spike (panic)
            price_at_bb_lower and  # Price at BB lower band
            bull_trend_4h and  # 4h trend bullish (trade with HTF)
            (zscore_oversold or volume_spike)  # Additional confirmation
        )
        
        # SHORT ENTRY: Mirror of long (only when 4h trend bearish)
        short_conditions = (
            vol_spike and  # Volatility spike (FOMO)
            price_at_bb_upper and  # Price at BB upper band
            bear_trend_4h and  # 4h trend bearish (trade with HTF)
            (zscore_overbought or volume_spike)  # Additional confirmation
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
            vol_spike_active = True
        
        if short_conditions:
            new_signal = -position_size
            vol_spike_active = True
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
                    vol_spike_active = False
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
                    vol_spike_active = False
        
        # === VOLATILITY NORMALIZATION EXIT ===
        # Exit when volatility normalizes (vol crush after panic)
        if in_position and vol_normal and vol_spike_active:
            new_signal = 0.0
            vol_spike_active = False
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based, stoploss, or vol normalization)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals