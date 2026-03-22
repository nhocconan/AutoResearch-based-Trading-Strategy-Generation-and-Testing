#!/usr/bin/env python3
"""
Experiment #020: 1h Primary + 4h/12h HTF — Fisher Transform Vol Reversion

Hypothesis: Previous strategies failed due to wrong regime detection and excessive trades.
This strategy combines proven edges for BTC/ETH in bear/range markets:

1. 12h HMA(21) for MAJOR trend bias (only trade WITH 12h trend)
2. 4h HMA(21) for INTERMEDIATE confirmation
3. Ehlers Fisher Transform(9) for reversal entries (crosses -1.5/+1.5)
4. ATR ratio(7/30) > 1.8 for vol spike confirmation (panic/reversal zones)
5. Bollinger %B < 0.1 or > 0.9 for extreme position (mean reversion setup)
6. Session filter (8-20 UTC only - high liquidity)
7. Volume > 0.8x 20-bar avg (confirm genuine moves)
8. ATR(14) trailing stoploss at 2.5x

Why this should work:
- Fisher Transform catches reversals in bear rallies (proven in 2022 crash)
- ATR ratio detects vol spikes = panic bottoms/tops (mean reversion edge)
- Bollinger %B extremes = oversold/overbought confirmation
- 12h/4h trend filter prevents counter-trend trades
- Session + volume filters reduce whipsaws
- Strict confluence = 30-60 trades/year (fee drag controlled)
- Discrete sizing (0.25) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (0.0, ±0.25)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_volspike_bb_4h12h_hma_v1"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals when Fisher crosses extreme levels (-1.5, +1.5).
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high + low) / 2
    typical_s = pd.Series(typical)
    
    # Highest high and lowest low over period
    hh = typical_s.rolling(window=period, min_periods=period).max()
    ll = typical_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to -1 to +1 range
    normalized = (typical - ll) / (hh - ll + 1e-10) * 2 - 1
    normalized = np.clip(normalized, -0.999, 0.999)  # Prevent division by zero
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = fisher.fillna(0).values
    
    return fisher

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands and %B position."""
    close_s = pd.Series(close)
    
    # Middle band (SMA)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    
    # Standard deviation
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    # Upper and lower bands
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    # %B position (0 = lower band, 1 = upper band)
    band_width = upper - lower + 1e-10
    percent_b = (close - lower) / band_width
    percent_b = np.nan_to_num(percent_b, nan=0.5)
    percent_b = np.clip(percent_b, 0, 1)
    
    return middle, upper, lower, percent_b

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher = calculate_fisher_transform(high, low, 9)
    bb_middle, bb_upper, bb_lower, percent_b = calculate_bollinger_bands(close, 20, 2.5)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for vol spike detection
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    # Fisher transform tracking for crosses
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(fisher[i]) or np.isnan(percent_b[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(atr_ratio[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            # If in position, can hold through off-session
            # If not in position, don't enter
            if not in_position:
                signals[i] = 0.0
                prev_fisher = fisher[i]
                continue
        
        # === 12H TREND BIAS (MAJOR) ===
        # Price above 12h HMA = bullish bias (prefer longs)
        # Price below 12h HMA = bearish bias (prefer shorts)
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === VOL SPIKE DETECTION (ATR ratio > 1.8) ===
        # High vol = panic/reversal zone (mean reversion opportunity)
        vol_spike = atr_ratio[i] > 1.8
        
        # === BOLLINGER %B EXTREMES ===
        # %B < 0.1 = extremely oversold (long opportunity)
        # %B > 0.9 = extremely overbought (short opportunity)
        bb_oversold = percent_b[i] < 0.1
        bb_overbought = percent_b[i] > 0.9
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        # Short: Fisher crosses below +1.5 from above
        fisher_long_cross = (prev_fisher < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (prev_fisher > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow entry when Fisher is at extreme (even without cross)
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - require 4+ confluence factors
        # Must have: 12h bullish + (4h bullish OR vol_spike) + (fisher cross OR extreme) + (bb oversold OR vol_spike)
        long_confidence = 0
        if trend_12h_bullish:
            long_confidence += 1
        if trend_4h_bullish or vol_spike:
            long_confidence += 1
        if fisher_long_cross or fisher_extreme_long:
            long_confidence += 1
        if bb_oversold or vol_spike:
            long_confidence += 1
        if volume_ok:
            long_confidence += 1
        
        if long_confidence >= 4:
            new_signal = current_size
        
        # SHORT ENTRIES - require 4+ confluence factors
        short_confidence = 0
        if trend_12h_bearish:
            short_confidence += 1
        if trend_4h_bearish or vol_spike:
            short_confidence += 1
        if fisher_short_cross or fisher_extreme_short:
            short_confidence += 1
        if bb_overbought or vol_spike:
            short_confidence += 1
        if volume_ok:
            short_confidence += 1
        
        if short_confidence >= 4:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~12 days on 1h), allow weaker entry (3 confluence)
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and trend_4h_bullish and (fisher_extreme_long or bb_oversold):
                new_signal = current_size * 0.8
            elif trend_12h_bearish and trend_4h_bearish and (fisher_extreme_short or bb_overbought):
                new_signal = -current_size * 0.8
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_12h_bearish and fisher[i] > 1.0:
                trend_reversal = True
            if position_side < 0 and trend_12h_bullish and fisher[i] < -1.0:
                trend_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 1.5:
                fisher_exit = True
            if position_side < 0 and fisher[i] < -1.5:
                fisher_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or fisher_exit:
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
        prev_fisher = fisher[i]
    
    return signals