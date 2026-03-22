#!/usr/bin/env python3
"""
Experiment #086: 12h Primary + 1d HTF — Fisher Transform + Volume Breakout + KAMA Trend

Hypothesis: Connors RSI worked moderately well (#076 Sharpe=0.220) but Fisher Transform
captures reversals more precisely in bear/range markets. Combined with volume confirmation
and KAMA (adaptive MA that adjusts to volatility), this should improve entry timing.

Strategy Logic:
1. EHLERS FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. VOLUME CONFIRMATION: Entry volume > 1.5x 20-bar avg volume (confirms breakout validity)
3. 1d KAMA(10): Adaptive trend filter - only long if price > KAMA, only short if price < KAMA
4. DONCHIAN(20) BREAKOUT: Entry trigger - long on upper breakout, short on lower breakout
5. CHOPPINESS INDEX (14): Meta-filter - avoid entries when CHOP 45-55 (transitional/noise)
6. ATR(14) stoploss: 2.5x trailing stop
7. Position size: 0.28 discrete (slightly lower than 0.30 for safety)

Why this should beat #076:
- Fisher Transform is purpose-built for reversal detection (Ehlers research)
- Volume filter prevents false breakouts (common issue on 12h)
- KAMA adapts to market volatility better than HMA in choppy conditions
- Donchian provides clear breakout levels (objective entry trigger)
- Different from Connors RSI approach = diversification of signal types

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_vol_kama_donchian_1d_v1"
timeframe = "12h"
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
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long signal: Fisher crosses above -1.5
    Short signal: Fisher crosses below +1.5
    """
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Normalize price over lookback period
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    # Normalize to -1 to +1 range
    normalized = 2.0 * (typical - lowest) / price_range - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    Adjusts smoothing based on market volatility/noise.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    Higher ER = trending (less smoothing), Lower ER = choppy (more smoothing)
    """
    close_s = pd.Series(close)
    
    # Net change over period
    net_change = np.abs(close_s - close_s.shift(period))
    
    # Sum of absolute changes (volatility)
    abs_changes = np.abs(close_s.diff())
    volatility = abs_changes.rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio
    er = net_change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate current volume vs rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_avg == 0, 1e-10, vol_avg)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d_10 = calculate_kama(df_1d['close'].values, 10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_10_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_10)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
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
        
        if np.isnan(kama_1d_10_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 1D KAMA TREND BIAS (MAJOR) ===
        # Price above 1d KAMA = bullish bias (prefer longs)
        # Price below 1d KAMA = bearish bias (prefer shorts)
        price_above_1d_kama = close[i] > kama_1d_10_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_10_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # Avoid trading in transitional markets (CHOP 45-55 = noise)
        is_clear_regime = chop_14[i] < 45 or chop_14[i] > 55
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === DONCHIAN BREAKOUT TRIGGER ===
        # Long: Price breaks above Donchian upper
        # Short: Price breaks below Donchian lower
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        # Volume must be > 1.5x average to confirm breakout validity
        volume_confirmed = vol_ratio[i] > 1.5
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets
        if not is_clear_regime:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Requires: 1d KAMA bullish + Fisher reversal OR Donchian breakout + Volume confirmation
        if price_above_1d_kama and is_clear_regime:
            # Fisher reversal entry (mean reversion in uptrend)
            if fisher_cross_up:
                new_signal = current_size
            # Donchian breakout entry (trend continuation)
            elif donchian_breakout_up and volume_confirmed:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Requires: 1d KAMA bearish + Fisher reversal OR Donchian breakout + Volume confirmation
        if price_below_1d_kama and is_clear_regime:
            # Fisher reversal entry (mean reversion in downtrend)
            if fisher_cross_down:
                new_signal = -current_size
            # Donchian breakout entry (trend continuation)
            elif donchian_breakout_down and volume_confirmed:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~50 days on 12h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if price_above_1d_kama and fisher[i] < -1.0:
                new_signal = current_size * 0.5
            elif price_below_1d_kama and fisher[i] > 1.0:
                new_signal = -current_size * 0.5
        
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
        # Exit if 1d Kama trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if price crosses below 1d KAMA
            if position_side > 0 and price_below_1d_kama:
                trend_reversal = True
            # Exit short if price crosses above 1d KAMA
            if position_side < 0 and price_above_1d_kama:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
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