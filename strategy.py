#!/usr/bin/env python3
"""
Experiment #005: 1h Primary + 4h/1d HTF — Vol Spike Reversion + Fisher Transform

Hypothesis: Previous Connors+Chop strategies failed because they're too common.
This strategy uses LESS COMMON indicators with proven edge in bear/range markets:

1. EHLERS FISHER TRANSFORM (period=9): Catches reversals in bear rallies
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Superior to RSI for crypto mean reversion

2. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol
   - Enter when vol spike + price at Bollinger Band extreme
   - Exit when vol normalizes (ATR ratio < 1.3)
   - Captures "vol crush" after panic selling

3. 4h/1d HMA(21) for TREND BIAS: Only trade WITH HTF trend
   - Prevents counter-trend mean reversion in strong trends
   - 1d for major bias, 4h for intermediate confirmation

4. ASYMMETRIC REGIME FILTER:
   - ADX > 25 + price < SMA50 = bear regime (only short retraces)
   - ADX < 20 = range regime (mean revert at BB bounds)
   - Hysteresis: enter at 25, exit at 18

5. SESSION FILTER (8-20 UTC): Avoid low-liquidity whipsaws

6. VOLUME CONFIRMATION: Volume > 0.8x 20-bar avg

Why this should work:
- Fisher Transform has better reversal detection than RSI in crypto
- Vol spike reversion captures panic bottoms (2022 crash, 2025 bear)
- HTF trend filter prevents deadly counter-trend trades
- Asymmetric regime adapts to market conditions
- 1h timeframe with strict filters = 30-60 trades/year target

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.30 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_volspike_4h1d_hma_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    for better reversal detection.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    3. Fisher: 0.5 * ln((1 + value) / (1 - value))
    4. Signal line: EMA of Fisher
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high + low) / 2
    typical_s = pd.Series(typical)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to -1 to +1
    normalized = (typical - ll) / (hh - ll).replace(0, np.nan) * 2 - 1
    normalized = normalized.clip(-0.999, 0.999)  # Prevent ln(0)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = fisher.replace([np.inf, -np.inf], np.nan)
    
    # Signal line (EMA of Fisher)
    fisher_s = pd.Series(fisher)
    signal = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return fisher.values, signal.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return sma.values, upper.values, lower.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    plus_di = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI values
    plus_di = (plus_di / atr).replace([np.inf, -np.inf], 0) * 100
    minus_di = (minus_di / atr).replace([np.inf, -np.inf], 0) * 100
    
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.5)
    adx = calculate_adx(high, low, close, 14)
    sma_50 = calculate_sma(close, 50)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Vol spike ratio
    vol_spike_ratio = atr_7 / atr_30
    vol_spike_ratio = np.nan_to_num(vol_spike_ratio, nan=1.0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    # Track Fisher crossover state
    prev_fisher_cross_long = False
    prev_fisher_cross_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(adx[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            if not in_position:
                signals[i] = 0.0
                continue
        
        # === 1D TREND BIAS (MAJOR) ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === REGIME DETECTION ===
        # ADX > 25 + price < SMA50 = bear regime (only short retraces)
        # ADX < 20 = range regime (mean revert at BB bounds)
        is_bear_regime = adx[i] > 25 and close[i] < sma_50[i]
        is_range_regime = adx[i] < 20
        is_trend_regime = adx[i] > 25
        
        # === VOL SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 2.0 = extreme volatility (panic)
        vol_spike = vol_spike_ratio[i] > 2.0
        vol_normal = vol_spike_ratio[i] < 1.3
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (fisher_signal[i] > -1.5) and (fisher_signal[i-1] <= -1.5)
        fisher_cross_short = (fisher_signal[i] < 1.5) and (fisher_signal[i-1] >= 1.5)
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in high vol
        if vol_spike:
            current_size = BASE_SIZE * 0.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        # Path 1: Vol spike reversion (panic bottom)
        # Requires: vol spike + at BB lower + HTF bullish bias
        if vol_spike and at_bb_lower:
            if trend_1d_bullish or trend_4h_bullish:
                if volume_ok:
                    new_signal = current_size
        
        # Path 2: Fisher reversal in range regime
        # Requires: Fisher cross long + range regime + HTF not strongly bearish
        if fisher_cross_long and is_range_regime:
            if not trend_1d_bearish:
                if volume_ok:
                    new_signal = current_size
        
        # Path 3: Fisher reversal + vol spike (strongest signal)
        if fisher_cross_long and vol_spike and at_bb_lower:
            new_signal = current_size * 1.2  # Slightly larger for strong confluence
            new_signal = min(new_signal, 0.35)  # Cap at max
        
        # SHORT ENTRIES - Multiple confluence paths
        # Path 1: Vol spike reversion (panic top)
        if vol_spike and at_bb_upper:
            if trend_1d_bearish or trend_4h_bearish:
                if volume_ok:
                    new_signal = -current_size
        
        # Path 2: Fisher reversal in range regime
        if fisher_cross_short and is_range_regime:
            if not trend_1d_bullish:
                if volume_ok:
                    new_signal = -current_size
        
        # Path 3: Bear regime short (only short retraces)
        if is_bear_regime:
            if fisher_cross_short and trend_4h_bearish:
                new_signal = -current_size
        
        # Path 4: Fisher reversal + vol spike (strongest signal)
        if fisher_cross_short and vol_spike and at_bb_upper:
            new_signal = -current_size * 1.2
            new_signal = max(new_signal, -0.35)  # Cap at max
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~12 days on 1h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and fisher_signal[i] < -1.0:
                new_signal = current_size * 0.6
            elif trend_1d_bearish and trend_4h_bearish and fisher_signal[i] > 1.0:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === VOL NORMALIZATION EXIT ===
        # Exit when vol spike normalizes (vol crush complete)
        vol_exit = False
        if in_position and vol_spike_ratio[i-1] > 2.0 and vol_normal:
            vol_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and fisher_signal[i] > 1.0:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and fisher_signal[i] < -1.0:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or vol_exit or trend_reversal:
            new_signal = 0.0
        
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