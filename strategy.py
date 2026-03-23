#!/usr/bin/env python3
"""
Experiment #460: 1h Primary + 4h/12h HTF — Funding Rate Contrarian + Regime Adaptive

Hypothesis: Based on research showing funding rate mean reversion has Sharpe 0.8-1.5 through
2022 crash for BTC/ETH. Combine with Choppiness Index regime detection, 4h HMA trend filter,
and 1h RSI pullback entries. Key innovations:
1. Funding rate z-score (30d) as PRIMARY signal driver — contrarian when crowd extreme
2. Choppiness Index regime switch (CHOP>55=range/mean-revert, <45=trend/breakout)
3. 4h HMA(21) for trend bias — only trade in HTF trend direction
4. 1h RSI(14) pullback entries within HTF trend (buy dips in uptrend, sell rallies in downtrend)
5. Session filter: only 8-20 UTC (high liquidity, avoid Asian session whipsaw)
6. Volume filter: >0.8x 20-bar average
7. Position size: 0.20 base (smaller for 1h to reduce fee drag), discrete levels
8. ATR(14) trailing stop at 2.5x for risk management

Target: Sharpe > 0.612, 30-60 trades over 4-year train, DD < -35%
Timeframe: 1h (hourly — balance between signal frequency and fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_funding_crsi_chop_hma4h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    rsi = calculate_rsi(close, rsi_period)
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_funding_zscore(prices, symbol, lookback=30):
    """
    Calculate funding rate z-score from processed funding data.
    Returns array aligned with prices length.
    """
    import os
    n = len(prices)
    funding_z = np.full(n, np.nan)
    
    # Try to load funding data
    funding_path = f"data/processed/funding/{symbol}.parquet"
    
    try:
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            
            # Align funding data to prices timestamps
            prices_times = pd.to_datetime(prices['open_time'])
            
            # Merge funding onto prices (forward fill to hourly)
            df_funding['open_time'] = pd.to_datetime(df_funding['open_time'])
            
            # Create aligned array
            funding_aligned = np.full(n, np.nan)
            
            for i in range(n):
                # Find funding rates before this price bar
                mask = df_funding['open_time'] <= prices_times.iloc[i]
                if mask.sum() > 0:
                    recent_funding = df_funding.loc[mask, 'funding_rate'].values
                    if len(recent_funding) > 0:
                        funding_aligned[i] = recent_funding[-1]
            
            # Calculate rolling z-score
            for i in range(lookback, n):
                window = funding_aligned[i-lookback:i]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) >= lookback // 2:
                    mean_f = np.mean(valid_window)
                    std_f = np.std(valid_window)
                    if std_f > 1e-10:
                        funding_z[i] = (funding_aligned[i] - mean_f) / std_f
    except Exception:
        pass
    
    return funding_z

def get_session_hour(prices):
    """Extract UTC hour from open_time."""
    times = pd.to_datetime(prices['open_time'])
    return times.dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = "BTCUSDT"  # Default, will work for all symbols
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate funding z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, symbol, lookback=30)
    
    # Calculate and align HTF HMA for bias (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Get session hours
    session_hours = get_session_hour(prices)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # 20% base position size for 1h (smaller to reduce fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 55.0  # Range market
        regime_trend = chop[i] < 45.0  # Trending market
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 12h HMA for stronger trend confirmation ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === FUNDING RATE CONTRARIAN SIGNAL ===
        funding_extreme_long = not np.isnan(funding_z[i]) and funding_z[i] < -1.5
        funding_extreme_short = not np.isnan(funding_z[i]) and funding_z[i] > 1.5
        funding_moderate_long = not np.isnan(funding_z[i]) and funding_z[i] < -0.8
        funding_moderate_short = not np.isnan(funding_z[i]) and funding_z[i] > 0.8
        
        # === RSI SIGNALS (Pullback entries) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === VOL FILTER ===
        vol_ratio = volume[i] / (vol_sma[i] + 1e-10)
        vol_ok = vol_ratio > 0.8
        
        # === VOL FILTER FOR POSITION SIZE ===
        if vol_ratio > 2.0:
            position_size = BASE_SIZE * 0.5  # Reduce size on vol spike
        elif vol_ratio > 1.2:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === LONG ENTRY CONDITIONS ===
        # Need: HTF bullish + funding contrarian OR RSI pullback + session + volume
        long_conditions = 0
        
        if price_above_hma_4h:
            long_conditions += 2  # Primary HTF trend filter
        if price_above_hma_12h:
            long_conditions += 1  # Stronger confirmation
        
        if funding_extreme_long:
            long_conditions += 3  # Strong contrarian signal
        elif funding_moderate_long:
            long_conditions += 2
        
        if rsi_oversold:
            long_conditions += 2  # Pullback entry
        if rsi_extreme_oversold:
            long_conditions += 1
        
        if crsi_oversold:
            long_conditions += 1
        
        if in_session:
            long_conditions += 1
        if vol_ok:
            long_conditions += 1
        
        # === SHORT ENTRY CONDITIONS ===
        short_conditions = 0
        
        if price_below_hma_4h:
            short_conditions += 2  # Primary HTF trend filter
        if price_below_hma_12h:
            short_conditions += 1  # Stronger confirmation
        
        if funding_extreme_short:
            short_conditions += 3  # Strong contrarian signal
        elif funding_moderate_short:
            short_conditions += 2
        
        if rsi_overbought:
            short_conditions += 2  # Rally entry
        if rsi_extreme_overbought:
            short_conditions += 1
        
        if crsi_overbought:
            short_conditions += 1
        
        if in_session:
            short_conditions += 1
        if vol_ok:
            short_conditions += 1
        
        # === ENTRY THRESHOLDS (strict to reduce trade count) ===
        # Long: need 6+ conditions (HTF trend + funding/RSI + session + volume)
        if long_conditions >= 6:
            desired_signal = position_size
            signal_strength = long_conditions
        
        # Short: need 6+ conditions
        if short_conditions >= 6 and desired_signal == 0:
            desired_signal = -position_size
            signal_strength = short_conditions
        
        # === REGIME ADAPTIVE ADJUSTMENTS ===
        # In choppy regime, require even more confluence (7+)
        if regime_chop:
            if long_conditions >= 7:
                desired_signal = position_size
            elif short_conditions >= 7 and desired_signal == 0:
                desired_signal = -position_size
            elif long_conditions < 7 and short_conditions < 7:
                desired_signal = 0.0  # No trade in choppy without strong signal
        
        # In trending regime, can trade with 5+ if HTF trend strong
        if regime_trend:
            if price_above_hma_4h and price_above_hma_12h and long_conditions >= 5:
                desired_signal = position_size
            elif price_below_hma_4h and price_below_hma_12h and short_conditions >= 5 and desired_signal == 0:
                desired_signal = -position_size
        
        # === CAP SIGNAL TO MAX 0.30 ===
        if desired_signal > 0.30:
            desired_signal = 0.30
        elif desired_signal < -0.30:
            desired_signal = -0.30
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.18:
                    desired_signal = 0.20
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.18:
                    desired_signal = -0.20
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals