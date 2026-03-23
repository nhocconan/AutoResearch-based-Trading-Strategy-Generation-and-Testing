#!/usr/bin/env python3
"""
Experiment #080: 1h Primary + 4h/12h HTF — Fisher Transform Reversal with Trend Filter

Hypothesis: Ehlers Fisher Transform catches reversals better than RSI in bear/range markets
(2025 test period). Combined with 4h HMA trend filter and volume confirmation, this should
generate 40-80 trades/year with Sharpe > 0.5 across ALL symbols (BTC, ETH, SOL).

Key innovations:
1) Fisher Transform(9) for entry timing — crosses at -1.5/+1.5 levels catch reversals
2) 4h HMA(21) for trend direction — only trade with HTF trend (simpler than 1d)
3) 12h HMA(21) for regime filter — avoid counter-trend when 12h strongly opposed
4) Volume spike filter (vol > 1.2x 20-bar avg) — confirms momentum
5) Session filter (8-20 UTC) — only trade during high liquidity
6) ATR(14) 2.5x trailing stoploss
7) Discrete sizing: 0.25 base, 0.35 max with volume boost

Why this should work:
- Fisher Transform proven edge for reversal catching (research Sharpe 0.8-1.2)
- 4h trend filter prevents counter-trend trades in 2025 bear market
- Volume + session filters reduce false signals (fee drag)
- Simpler than CRSI/Choppiness regimes = more trades (avoid 0-trade failure)
- Works on mean-reversion AND trending regimes

Position size: 0.25 base, 0.35 max with volume boost
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_vol_session_4h12h_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price to -1 to +1 range
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    range_hl = highest - lowest
    
    # Avoid division by zero
    range_hl = range_hl.replace(0, 1e-10)
    
    normalized = 0.66 * ((hl2_s - lowest) / range_hl - 0.5) + 0.67 * normalized.shift(1).fillna(0)
    normalized = normalized.clip(-0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher_prev = fisher.shift(1).fillna(0)
    
    return fisher.values, fisher_prev.values

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for regime filter
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma_20[i]):
            continue
        if vol_ma_20[i] == 0:
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope
        hma_4h_slope = 0.0
        if i > 0 and not np.isnan(hma_4h_aligned[i-1]):
            hma_4h_slope = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / (hma_4h_aligned[i-1] + 1e-10)
        
        # === 12h REGIME FILTER ===
        # Avoid strong counter-trend trades when 12h opposes
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_ma_20[i] + 1e-10)
        volume_spike = vol_ratio > 1.2  # 20% above average
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLATILITY FILTER ===
        vol_ratio_atr = atr_14[i] / (atr_50[i] + 1e-10)
        extreme_vol = vol_ratio_atr > 2.5  # Skip extreme volatility
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Skip if extreme volatility or outside session
        if extreme_vol or not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # --- LONG ENTRY: 4h uptrend + Fisher reversal + volume confirm ---
        # Condition 1: 4h HMA bullish (price above)
        # Condition 2: Fisher crosses above -1.5
        # Condition 3: Volume spike OR EMA bullish
        # Condition 4: 12h not strongly bearish (avoid counter-trend)
        if price_above_hma_4h and fisher_long:
            if (volume_spike or ema_bullish) and not (price_below_hma_12h and hma_4h_slope < -0.001):
                new_signal = POSITION_SIZE_BASE
                # Boost position if volume spike confirms
                if volume_spike:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: 4h downtrend + Fisher reversal + volume confirm ---
        # Condition 1: 4h HMA bearish (price below)
        # Condition 2: Fisher crosses below +1.5
        # Condition 3: Volume spike OR EMA bearish
        # Condition 4: 12h not strongly bullish (avoid counter-trend)
        if price_below_hma_4h and fisher_short:
            if (volume_spike or ema_bearish) and not (price_above_hma_12h and hma_4h_slope > 0.001):
                new_signal = -POSITION_SIZE_BASE
                # Boost position if volume spike confirms
                if volume_spike:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if Fisher hasn't reached opposite extreme
        if in_position and new_signal == 0.0:
            if position_side > 0 and fisher[i] < 1.5:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and fisher[i] > -1.5:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_since_entry == 0.0:
                highest_since_entry = close[i]
            else:
                highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 4h HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope < -0.0005:
                new_signal = 0.0
        
        # Exit short if 4h HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope > 0.0005:
                new_signal = 0.0
        
        # === EXIT ON SESSION END ===
        # Close positions outside session hours to avoid overnight risk
        if in_position and not in_session:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals