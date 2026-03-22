#!/usr/bin/env python3
"""
Experiment #288: 1d HMA Trend with 4h Bias and Dual Entry Modes

Hypothesis: Previous 1d strategies failed due to (1) too-slow HTF (1w), 
(2) restrictive volume filters, (3) single entry mode. This strategy uses:

1. 1d HMA(16/48) crossover for primary trend signal - faster than Donchian
2. 4h HMA(21) for directional bias - more responsive than 1w, catches trends earlier
3. DUAL entry modes to ensure >=10 trades:
   - Trend continuation: price > both HMAs + momentum confirmation
   - Mean reversion: RSI(7) extremes when aligned with 4h trend
4. NO volume filter - was filtering out valid signals in #276, #282
5. 2.5*ATR trailing stoploss - tighter than 3.0*ATR for better risk control
6. Asymmetric sizing: 0.30 normal, 0.20 in high vol, 0.35 in strong trend

Why this might beat previous 1d attempts:
- 4h HTF is faster/more responsive than 1w (used in #276, #282 which failed)
- Dual entry modes = more trade opportunities (trend + mean reversion)
- No volume filter = fewer missed signals
- RSI(7) extremes catch pullbacks that Donchian miss

Timeframe: 1d (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_dual_entry_4h_bias_rsi_atr_v1"
timeframe = "1d"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_momentum(close, period=10):
    """Calculate price momentum (rate of change)."""
    close_s = pd.Series(close)
    mom = close_s.pct_change(periods=period)
    return mom.fillna(0.0).values

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
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    ema_50 = calculate_ema(close, 50)
    momentum = calculate_momentum(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_REDUCED = 0.20
    SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = directional bias (softer filter than 1d HMA)
        bull_bias_4h = close[i] > hma_4h_aligned[i]
        bear_bias_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D TREND SIGNAL ===
        # HMA crossover for primary trend
        hma_crossover_long = hma_fast[i] > hma_slow[i]
        hma_crossover_short = hma_fast[i] < hma_slow[i]
        
        # Price above/below both HMAs = strong trend
        strong_bull = close[i] > hma_fast[i] and close[i] > hma_slow[i]
        strong_bear = close[i] < hma_fast[i] and close[i] < hma_slow[i]
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === MODE 1: TREND CONTINUATION ===
        # Enter when trend is strong and aligned with 4h bias
        trend_long = (
            strong_bull and  # Price above both HMAs
            hma_crossover_long and  # Fast HMA above slow
            bull_bias_4h and  # 4h bias bullish
            momentum[i] > 0.0  # Positive momentum
        )
        
        trend_short = (
            strong_bear and  # Price below both HMAs
            hma_crossover_short and  # Fast HMA below slow
            bear_bias_4h and  # 4h bias bearish
            momentum[i] < 0.0  # Negative momentum
        )
        
        # === MODE 2: MEAN REVERSION PULLBACK ===
        # Enter on RSI extremes when aligned with 4h trend
        # Long: RSI(7) < 30 oversold + 4h bullish bias + price > EMA50
        mr_long = (
            rsi_7[i] < 30 and  # Oversold on fast RSI
            bull_bias_4h and  # 4h bias still bullish
            close[i] > ema_50[i]  # Above long-term average
        )
        
        # Short: RSI(7) > 70 overbought + 4h bearish bias + price < EMA50
        mr_short = (
            rsi_7[i] > 70 and  # Overbought on fast RSI
            bear_bias_4h and  # 4h bias still bearish
            close[i] < ema_50[i]  # Below long-term average
        )
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Priority: trend signals over mean reversion
        if trend_long:
            new_signal = position_size
        elif trend_short:
            new_signal = -position_size
        elif mr_long and not in_position:
            # Mean reversion only if not already positioned
            new_signal = position_size * 0.8  # Slightly smaller for MR
        elif mr_short and not in_position:
            new_signal = -position_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_bias_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
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
        
        signals[i] = new_signal
    
    return signals