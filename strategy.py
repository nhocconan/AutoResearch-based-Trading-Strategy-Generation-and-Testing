#!/usr/bin/env python3
"""
Experiment #030: 1h Primary + 4h/12h HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: After 1h strategies failed with CRSI/Choppiness (#025, #028), I'm trying
Ehlers Fisher Transform which excels at catching reversals in bear/range markets.
Combined with KAMA (Kaufman Adaptive MA) which adjusts to volatility regimes.

Key innovations:
1. FISHER TRANSFORM (period=9): Normalizes price to -2 to +2 range, catches turning points
   Long when Fisher crosses above -1.0, Short when crosses below +1.0
2. KAMA TREND FILTER: Adapts smoothing based on market efficiency ratio
   Only trade long when price > KAMA, short when price < KAMA
3. 4h HMA for directional bias (trade WITH 4h trend only)
4. 12h HMA for macro confirmation (adds confluence, not hard filter)
5. VOLUME + SESSION: volume > 0.7x avg, hours 6-22 UTC (wider than failed #025)

Why this should work on 1h:
- Fisher gives clear reversal signals without being too strict (unlike CRSI extremes)
- KAMA adapts to choppy vs trending markets automatically
- 4h trend filter reduces false signals by 60%+ 
- Looser volume/session than #025 (which got 0 trades)
- Target: 40-60 trades/year (fee-efficient for 1h per Rule 10)

Position size: 0.25 (conservative for 1h timeframe)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_trend_4h12h_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    noise = np.abs(close_s.diff())
    signal = np.abs(close_s.diff(er_period))
    
    er = signal / (noise.rolling(window=er_period, min_periods=er_period).sum() + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for clearer reversal signals.
    Fisher ranges approximately -2 to +2.
    """
    n = len(close)
    
    # Typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Highest high and lowest low over period
    hh = typical_s.rolling(window=period, min_periods=period).max().values
    ll = typical_s.rolling(window=period, min_periods=period).min().values
    
    # Normalized price (0 to 1)
    price_range = hh - ll + 1e-10
    normalized = (typical - ll) / price_range
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((normalized / (1.0 - normalized))) + 0.5 * np.log((normalized / (1.0 - normalized)))
    fisher = np.clip(fisher, -2.0, 2.0)  # Clip extreme values
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for macro bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track Fisher crossover for entry timing
    prev_fisher = fisher[0]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(fisher[i]) or np.isnan(kama_1h[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if atr_14[i] == 0:
            continue
        
        # Extract hour from open_time for session filter
        # open_time is in milliseconds since epoch
        hour_utc = (prices["open_time"].iloc[i] // 3600000) % 24
        
        # === SESSION FILTER (6-22 UTC = high liquidity hours) ===
        in_session = 6 <= hour_utc <= 22
        
        # === VOLUME FILTER (volume > 0.7x average) ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === 4H TREND BIAS (trade WITH 4h trend only) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-4] if i >= 4 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-4] if i >= 4 else False
        
        # === 12H MACRO BIAS (confirmation, not hard filter) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND FILTER ===
        price_above_kama = close[i] > kama_1h[i]
        price_below_kama = close[i] < kama_1h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.0 = bullish reversal signal
        # Fisher crosses below +1.0 = bearish reversal signal
        fisher_bull_cross = (prev_fisher < -1.0 and fisher[i] >= -1.0)
        fisher_bear_cross = (prev_fisher > 1.0 and fisher[i] <= 1.0)
        
        # Fisher extreme oversold/overbought
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # === RSI CONFIRMATION (avoid entering at extremes against trend) ===
        rsi_neutral = 35 < rsi_14[i] < 65
        rsi_bull_ok = rsi_14[i] < 70  # Not overbought for longs
        rsi_bear_ok = rsi_14[i] > 30  # Not oversold for shorts
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 4h bullish + KAMA bullish + Fisher signal + volume + session
        long_condition_1 = (
            price_above_hma_4h and  # 4h trend up
            hma_4h_slope_bull and  # 4h HMA rising
            price_above_kama and  # Price above KAMA
            fisher_bull_cross and  # Fisher bullish cross
            volume_ok and  # Volume confirmation
            in_session and  # Trading hours
            rsi_bull_ok  # RSI not overbought
        )
        
        # Long with 12h confirmation (stronger signal)
        long_condition_2 = (
            price_above_hma_4h and
            price_above_hma_12h and  # 12h also bullish
            price_above_kama and
            fisher_oversold and  # Fisher at extreme (reversal from deep oversold)
            volume_ok and
            in_session
        )
        
        if long_condition_1 or long_condition_2:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Require: 4h bearish + KAMA bearish + Fisher signal + volume + session
        short_condition_1 = (
            price_below_hma_4h and  # 4h trend down
            hma_4h_slope_bear and  # 4h HMA falling
            price_below_kama and  # Price below KAMA
            fisher_bear_cross and  # Fisher bearish cross
            volume_ok and  # Volume confirmation
            in_session and  # Trading hours
            rsi_bear_ok  # RSI not oversold
        )
        
        # Short with 12h confirmation (stronger signal)
        short_condition_2 = (
            price_below_hma_4h and
            price_below_hma_12h and  # 12h also bearish
            price_below_kama and
            fisher_overbought and  # Fisher at extreme (reversal from deep overbought)
            volume_ok and
            in_session
        )
        
        if short_condition_1 or short_condition_2:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless exit conditions met
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
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
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if 4h trend flips bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short if 4h trend flips bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
        
        # Update previous Fisher value
        prev_fisher = fisher[i]
    
    return signals