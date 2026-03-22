#!/usr/bin/env python3
"""
Experiment #225: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Session Filter

Hypothesis: After 224 experiments, RSI-based strategies are overused and failing.
Fisher Transform (Ehlers) is proven to catch reversals in bear/range markets better
than RSI, especially for BTC/ETH which failed simple trend strategies.

Key innovations:
1. FISHER TRANSFORM (period=9): Normalizes price to -2 to +2, catches reversals early
2. 4h HMA(21) for trend direction (HTF bias)
3. 1d HMA(21) for major regime filter (never trade against daily trend)
4. SESSION FILTER (8-20 UTC): Only trade during high liquidity hours
5. VOLUME CONFIRMATION: Volume > 0.8x 20-bar average
6. ATR(14) trailing stop at 2.5x for risk management

Why 1h timeframe:
- Balances trade frequency (30-60/year target) with signal quality
- Session filter naturally limits trades to liquid hours
- 4h/1d HTF provides trend context, 1h provides entry timing

Position sizing: 0.25 discrete (conservative for lower TF)
Target: 40-80 trades/year per symbol
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_session_4h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Better at catching reversals than RSI in bear/range markets.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1e-10, range_val)
    normalized = (hl2 - lowest) / range_val
    
    # Clamp to 0.001-0.999 to avoid log(0)
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-bar lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / np.where(vol_avg == 0, 1e-10, vol_avg)
    return ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time_array // (1000 * 3600)) % 24
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    volume_ratio = calculate_volume_ratio(volume, 20)
    utc_hour = get_hour_from_open_time(open_time)
    
    # 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, 21)
    hma_1h_slope = calculate_hma_slope(hma_1h_21, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    BASE_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(volume_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high liquidity hours
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume_ratio[i] > 0.8
        
        # === HTF TREND BIAS (4h) ===
        # 4h trend determines primary bias
        hma_4h_bullish = hma_4h_slope_aligned[i] > 0.10
        hma_4h_bearish = hma_4h_slope_aligned[i] < -0.10
        hma_4h_neutral = not hma_4h_bullish and not hma_4h_bearish
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === MAJOR REGIME FILTER (1d) ===
        # Never trade against daily trend
        hma_1d_bullish = hma_1d_slope_aligned[i] > 0.08
        hma_1d_bearish = hma_1d_slope_aligned[i] < -0.08
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (1h HMA) ===
        hma_1h_bullish = hma_1h_slope[i] > 0.15
        hma_1h_bearish = hma_1h_slope[i] < -0.15
        
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_bull_cross = (fisher_signal[i] < -1.2) and (fisher[i] > fisher_signal[i]) and (fisher[i] > -1.5)
        fisher_bear_cross = (fisher_signal[i] > 1.2) and (fisher[i] < fisher_signal[i]) and (fisher[i] < 1.5)
        
        # Extreme Fisher values (oversold/overbought)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # Fisher trending up/down
        fisher_rising = fisher[i] > fisher_signal[i]
        fisher_falling = fisher[i] < fisher_signal[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        
        # Path 1: Fisher bullish cross + 4h bullish + session + volume (primary)
        if fisher_bull_cross and hma_4h_bullish and in_session and volume_confirmed:
            long_score += 5
        
        # Path 2: Fisher bullish cross + 4h bullish + price above 4h HMA
        if fisher_bull_cross and hma_4h_bullish and price_above_4h_hma:
            long_score += 4
        
        # Path 3: Fisher oversold + 4h bullish + daily bullish (deep pullback in uptrend)
        if fisher_oversold and hma_4h_bullish and hma_1d_bullish:
            long_score += 4
        
        # Path 4: Fisher rising + 4h bullish + session + volume
        if fisher_rising and hma_4h_bullish and in_session and volume_confirmed and price_above_1h_hma:
            long_score += 3
        
        # Path 5: 4h bullish + daily bullish + price above both HMAs + Fisher rising
        if hma_4h_bullish and hma_1d_bullish and price_above_4h_hma and price_above_1d_hma and fisher_rising:
            long_score += 3
        
        # Path 6: Fisher bullish cross + daily bullish (regime-aligned reversal)
        if fisher_bull_cross and hma_1d_bullish and price_above_1d_hma:
            long_score += 3
        
        # Path 7: 4h bullish + Fisher oversold (buy the dip in uptrend)
        if hma_4h_bullish and fisher_oversold and price_above_4h_hma and bars_since_last_trade > 20:
            long_score += 2
        
        # Path 8: Session + volume + Fisher rising + price above 1h HMA
        if in_session and volume_confirmed and fisher_rising and price_above_1h_hma and bars_since_last_trade > 15:
            long_score += 2
        
        # Apply long signal based on score
        if long_score >= 5:
            new_signal = current_size
        elif long_score >= 4:
            new_signal = current_size
        elif long_score >= 3 and bars_since_last_trade > 10:
            new_signal = current_size * 0.8
        elif long_score >= 2 and bars_since_last_trade > 20:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Fisher bearish cross + 4h bearish + session + volume (primary)
        if fisher_bear_cross and hma_4h_bearish and in_session and volume_confirmed:
            short_score += 5
        
        # Path 2: Fisher bearish cross + 4h bearish + price below 4h HMA
        if fisher_bear_cross and hma_4h_bearish and price_below_4h_hma:
            short_score += 4
        
        # Path 3: Fisher overbought + 4h bearish + daily bearish (rally in downtrend)
        if fisher_overbought and hma_4h_bearish and hma_1d_bearish:
            short_score += 4
        
        # Path 4: Fisher falling + 4h bearish + session + volume
        if fisher_falling and hma_4h_bearish and in_session and volume_confirmed and price_below_1h_hma:
            short_score += 3
        
        # Path 5: 4h bearish + daily bearish + price below both HMAs + Fisher falling
        if hma_4h_bearish and hma_1d_bearish and price_below_4h_hma and price_below_1d_hma and fisher_falling:
            short_score += 3
        
        # Path 6: Fisher bearish cross + daily bearish (regime-aligned reversal)
        if fisher_bear_cross and hma_1d_bearish and price_below_1d_hma:
            short_score += 3
        
        # Path 7: 4h bearish + Fisher overbought (sell the rally in downtrend)
        if hma_4h_bearish and fisher_overbought and price_below_4h_hma and bars_since_last_trade > 20:
            short_score += 2
        
        # Path 8: Session + volume + Fisher falling + price below 1h HMA
        if in_session and volume_confirmed and fisher_falling and price_below_1h_hma and bars_since_last_trade > 15:
            short_score += 2
        
        # Apply short signal based on score
        if short_score >= 5:
            new_signal = -current_size
        elif short_score >= 4:
            new_signal = -current_size
        elif short_score >= 3 and bars_since_last_trade > 10:
            new_signal = -current_size * 0.8
        elif short_score >= 2 and bars_since_last_trade > 20:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 80 bars (~80 hours = 3+ days)
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and hma_1d_bullish and fisher_rising and price_above_1h_hma:
                new_signal = current_size * 0.4
            elif hma_4h_bearish and hma_1d_bearish and fisher_falling and price_below_1h_hma:
                new_signal = -current_size * 0.4
            elif fisher_oversold and price_above_1d_hma and bars_since_last_trade > 100:
                new_signal = current_size * 0.3
            elif fisher_overbought and price_below_1d_hma and bars_since_last_trade > 100:
                new_signal = -current_size * 0.3
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h turns strongly bearish
            if position_side > 0 and hma_4h_bearish and price_below_4h_hma:
                trend_reversal = True
            # Short position but 4h turns strongly bullish
            if position_side < 0 and hma_4h_bullish and price_above_4h_hma:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
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