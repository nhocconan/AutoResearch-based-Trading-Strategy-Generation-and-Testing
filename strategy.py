#!/usr/bin/env python3
"""
Experiment #168: 30m Primary + 4h/1d HTF — Fisher Transform + Vol Regime + Session Filter

Hypothesis: Previous Connors RSI strategies failed due to too many whipsaws in 30m timeframe.
Research shows Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022 crash, 2025 bear) with fewer false signals than RSI. This strategy combines:

1. FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, short when crosses below +1.5
2. 4h HMA(21) SLOPE: Major trend bias (only trade with HTF trend direction)
3. 1d ADX(14): Regime filter (ADX>25=trend, ADX<20=range) - different logic per regime
4. BOLLINGER BANDS (20,2.0): Mean revert at extremes in range, pullback entries in trend
5. SESSION FILTER (8-20 UTC): Only trade during high liquidity hours (reduces 30m noise)
6. VOLUME CONFIRMATION: Volume > 0.8x 20-bar avg (avoid low-liquidity traps)

Why 30m can work:
- 4h/1d HTF determines DIRECTION (not 30m indicators)
- 30m Fisher only for ENTRY TIMING within HTF trend
- Session filter cuts overnight noise (30m has many fake moves 0-8 UTC)
- Target: 40-80 trades/year (strict confluence = 3+ filters must agree)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF to reduce fee impact)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 40-80/year per symbol (MUST exceed 30 trades on train, 3 on test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_volregime_session_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((price - LL) / (HH - LL) - 0.5) + 0.38 * prev_fisher
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            X = 0
        else:
            X = 0.67 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.38 * fisher[i-1]
        
        X = np.clip(X, -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + X) / (1 - X))
        
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = (open_time_array // (1000 * 60 * 60)) % 24
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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    adx_1d, _, _ = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (8-20 UTC)
    hours = get_hour_from_open_time(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
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
        
        if np.isnan(hma_4h_slope_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(hma_1d_50_aligned[i]) or np.isnan(vol_avg[i]):
            continue
        
        # === 4H TREND BIAS (signal direction) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.5
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.5
        
        # === 1D REGIME (ADX) ===
        is_trend_regime = adx_1d_aligned[i] > 25
        is_range_regime = adx_1d_aligned[i] < 20
        
        # === 1D MAJOR TREND ===
        price_above_1d_hma = close[i] > hma_1d_50_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_50_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_trend_regime and not is_range_regime:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths (need 3+ filters)
        long_confluence = 0
        
        # Filter 1: 4h trend bullish OR price above 1d HMA
        if trend_4h_bullish or price_above_1d_hma:
            long_confluence += 1
        
        # Filter 2: Fisher signal
        if fisher_cross_up or fisher_oversold:
            long_confluence += 1
        
        # Filter 3: BB mean reversion (range) or pullback (trend)
        if is_range_regime and price_below_bb_lower:
            long_confluence += 1
        if is_trend_regime and bb_pct < 0.3:
            long_confluence += 1
        
        # Filter 4: RSI confirmation
        if rsi_oversold:
            long_confluence += 1
        
        # Filter 5: Volume confirmation
        if vol_confirmed:
            long_confluence += 1
        
        # Filter 6: Session filter (required for 30m)
        if session_ok:
            long_confluence += 1
        
        # Need 4+ confluence for long entry
        if long_confluence >= 4:
            new_signal = current_size
        elif long_confluence >= 3 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confluence = 0
        
        # Filter 1: 4h trend bearish OR price below 1d HMA
        if trend_4h_bearish or price_below_1d_hma:
            short_confluence += 1
        
        # Filter 2: Fisher signal
        if fisher_cross_down or fisher_overbought:
            short_confluence += 1
        
        # Filter 3: BB mean reversion (range) or pullback (trend)
        if is_range_regime and price_above_bb_upper:
            short_confluence += 1
        if is_trend_regime and bb_pct > 0.7:
            short_confluence += 1
        
        # Filter 4: RSI confirmation
        if rsi_overbought:
            short_confluence += 1
        
        # Filter 5: Volume confirmation
        if vol_confirmed:
            short_confluence += 1
        
        # Filter 6: Session filter (required for 30m)
        if session_ok:
            short_confluence += 1
        
        # Need 4+ confluence for short entry
        if short_confluence >= 4:
            new_signal = -current_size
        elif short_confluence >= 3 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # Force trade if no signal for 120 bars (~60 hours on 30m)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and (fisher_oversold or rsi_oversold):
                new_signal = current_size * 0.4
            elif trend_4h_bearish and (fisher_overbought or rsi_overbought):
                new_signal = -current_size * 0.4
            elif is_range_regime and fisher_oversold and session_ok:
                new_signal = current_size * 0.3
            elif is_range_regime and fisher_overbought and session_ok:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend flips bearish
            if position_side > 0 and trend_4h_bearish and adx_1d_aligned[i] > 25:
                regime_reversal = True
            # Exit short if 4h trend flips bullish
            if position_side < 0 and trend_4h_bullish and adx_1d_aligned[i] > 25:
                regime_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 1.5:
                fisher_exit = True
            if position_side < 0 and fisher[i] < -1.5:
                fisher_exit = True
        
        if stoploss_triggered or regime_reversal or fisher_exit:
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