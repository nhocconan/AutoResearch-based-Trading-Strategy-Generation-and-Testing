#!/usr/bin/env python3
"""
Experiment #009: 4h Vol Spike Mean Reversion with Daily Trend Filter

Hypothesis: Previous CRSI/Choppiness strategies failed because they tried mean
reversion in trending markets. Research shows VOL SPIKE REVERSION works best:
- ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol
- Enter when price < BB(20, 2.5) after vol spike (capitulation)
- Exit when ATR ratio normalizes < 1.3 (vol crush complete)
- 1d HMA(21) filter: only long if price > 1d HMA, only short if <
- Asymmetric sizing: 0.30 for high-confidence, 0.20 for moderate

Why this should work:
- Vol spike reversion reported Sharpe 0.8-1.5 through 2022 crash
- Captures "panic → recovery" pattern common in crypto
- Daily trend filter prevents counter-trend mean reversion (major failure mode)
- 4h TF targets 25-45 trades/year (optimal for this strategy)
- Works on BTC/ETH/SOL (all have vol spike patterns)

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_bb_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return atr.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending, ADX < 20 = ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Vol spike ratio
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    HIGH_CONF_SIZE = 0.30
    MED_CONF_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    vol_spike_active = False
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0  # Extreme vol (panic)
        vol_normal = vol_ratio[i] < 1.3  # Vol normalized
        
        # Track vol spike state
        if vol_spike:
            vol_spike_active = True
        elif vol_normal:
            vol_spike_active = False
        
        # === BOLLINGER BAND EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        bb_near_lower = close[i] < bb_mid[i] - 1.5 * (bb_upper[i] - bb_lower[i]) / 4
        bb_near_upper = close[i] > bb_mid[i] + 1.5 * (bb_upper[i] - bb_lower[i]) / 4
        
        # === ADX REGIME ===
        adx_trending = adx_14[i] > 25
        adx_ranging = adx_14[i] < 20
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_extreme_oversold = rsi_14[i] < 20
        rsi_extreme_overbought = rsi_14[i] > 80
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Vol spike + BB oversold + trend alignment
        long_confidence = 0
        long_score = 0
        
        # Primary trigger: Vol spike capitulation
        if vol_spike and bb_oversold:
            long_score += 2.5
            long_confidence = 1
        elif vol_spike_active and bb_near_lower:
            long_score += 1.5
            long_confidence = 0.7
        elif bb_oversold and rsi_extreme_oversold:
            long_score += 1.5
            long_confidence = 0.7
        
        # Trend alignment (bullish bias)
        if daily_bullish:
            long_score += 1.5
        elif price_above_sma200:
            long_score += 1.0
        
        # ADX regime (mean reversion works in ranging)
        if adx_ranging:
            long_score += 1.0
        elif not adx_trending:
            long_score += 0.5
        
        # RSI confirmation
        if rsi_oversold:
            long_score += 0.5
        if rsi_extreme_oversold:
            long_score += 0.5
        
        # Enter long if score >= 4.5 (strong confluence)
        if long_score >= 4.5:
            new_signal = HIGH_CONF_SIZE if long_confidence == 1 else MED_CONF_SIZE
        
        # SHORT ENTRY: Vol spike + BB overbought + trend alignment
        short_confidence = 0
        short_score = 0
        
        # Primary trigger: Vol spike capitulation
        if vol_spike and bb_overbought:
            short_score += 2.5
            short_confidence = 1
        elif vol_spike_active and bb_near_upper:
            short_score += 1.5
            short_confidence = 0.7
        elif bb_overbought and rsi_extreme_overbought:
            short_score += 1.5
            short_confidence = 0.7
        
        # Trend alignment (bearish bias)
        if daily_bearish:
            short_score += 1.5
        elif price_below_sma200:
            short_score += 1.0
        
        # ADX regime
        if adx_ranging:
            short_score += 1.0
        elif not adx_trending:
            short_score += 0.5
        
        # RSI confirmation
        if rsi_overbought:
            short_score += 0.5
        if rsi_extreme_overbought:
            short_score += 0.5
        
        # Enter short if score >= 4.5 (strong confluence)
        if short_score >= 4.5:
            new_signal = -HIGH_CONF_SIZE if short_confidence == 1 else -MED_CONF_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~17 days on 4h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if bb_oversold and rsi_oversold and daily_bullish:
                new_signal = MED_CONF_SIZE
            elif bb_overbought and rsi_overbought and daily_bearish:
                new_signal = -MED_CONF_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === VOL NORMALIZATION EXIT ===
        vol_exit = False
        if in_position and position_side != 0:
            # Exit when vol normalizes (vol crush complete)
            if vol_normal and vol_spike_active:
                vol_exit = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI goes overbought (mean reversion complete)
            if position_side > 0 and rsi_14[i] > 65:
                rsi_exit = True
            # Exit short if RSI goes oversold (mean reversion complete)
            if position_side < 0 and rsi_14[i] < 35:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if major trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if major trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or vol_exit or rsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals